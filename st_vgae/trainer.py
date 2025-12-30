import torch
import torch.optim as optim
from tqdm import tqdm
from .loss import LossFunction


class Trainer:
    def __init__(self, model, dataset, device='cuda:0',
                 lr=1e-3, weight_decay=1e-5,
                 beta_start=0.0, beta_end=0.35, n_epochs_kl_warmup=200,
                 alpha_l1=0.005,
                 prune_threshold=1e-4,
                 alpha_ortho=1.0,
                 addon_warmup_epochs=50):  # ✅ 新增参数：Addon 预热轮数

        self.model = model.to(device)
        self.device = device
        self.x_input = dataset.x_norm.to(device)
        self.x_raw = dataset.x_raw.to(device)
        self.sf = dataset.scale_factor.to(device)
        self.edge_index = dataset.edge_index.to(device)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = LossFunction()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_steps = n_epochs_kl_warmup
        self.alpha_l1 = alpha_l1
        self.alpha_addon = alpha_l1 * 50.0  # Addon 的惩罚通常比 Pathway 重一点
        self.alpha_ortho = alpha_ortho
        self.prune_threshold = prune_threshold

        # ✅ 保存预热轮数
        self.addon_warmup_epochs = addon_warmup_epochs

        self.logs = {'loss': [], 'recon': []}

    def train(self, n_epochs=300, log_interval=25):
        self.model.train()
        pbar = tqdm(range(n_epochs), desc="Training")

        for epoch in pbar:
            self.optimizer.zero_grad()
            current_beta = min(self.beta_end,
                               self.beta_start + (epoch / self.warmup_steps) * (self.beta_end - self.beta_start))

            # Forward
            mean, disp, pi, mu, log_var, z = self.model(self.x_input, self.edge_index, self.sf)

            # Loss
            total_loss, recon, kl, ortho = self.criterion(
                mean=mean, disp=disp, pi=pi, x=self.x_raw, mu=mu, log_var=log_var,
                current_beta=current_beta,
                decoder_layer=self.model.decoder,
                alpha_l1=self.alpha_l1,
                alpha_addon=self.alpha_addon,
                alpha_ortho=self.alpha_ortho
            )

            # Backward
            total_loss.backward()

            # ==========================================
            # ✅ 核心策略: Addon Warmup (锁死 Addon)
            # ==========================================
            if epoch < self.addon_warmup_epochs:
                # 如果处于预热期，强制把 Addon 的梯度“杀掉”
                # 这样优化器就无法更新 Addon 的权重，模型被迫去优化 Pathway
                if self.model.decoder.weight_addon is not None:
                    self.model.decoder.weight_addon.grad = None
            # ==========================================

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Pruning (软剪枝)
            self.prune_weights(threshold=self.prune_threshold)

            self.logs['loss'].append(total_loss.item())
            self.logs['recon'].append(recon.item())

            if (epoch + 1) % log_interval == 0:
                # 显示当前 Addon 状态
                status = "LOCKED" if epoch < self.addon_warmup_epochs else "ACTIVE"

                pbar.write(
                    f"Ep {epoch + 1:03d} [{status}] | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Recon: {recon.item():.4f} | "
                    f"Ortho: {ortho.item():.4f} | "
                    f"KL: {kl.item():.4f}"
                )

    def prune_weights(self, threshold):
        with torch.no_grad():
            w_path = self.model.decoder.weight_pathway
            mask = self.model.decoder.mask
            inverse_mask = (1.0 - mask).bool()
            # 把 Mask 外面绝对值很小的权重置为 0
            w_path.data[inverse_mask] = torch.where(
                torch.abs(w_path.data[inverse_mask]) < threshold,
                torch.tensor(0.0, device=self.device),
                w_path.data[inverse_mask]
            )
            if self.model.decoder.weight_addon is not None:
                w_addon = self.model.decoder.weight_addon
                w_addon.data = torch.where(
                    torch.abs(w_addon.data) < threshold,
                    torch.tensor(0.0, device=self.device),
                    w_addon.data
                )