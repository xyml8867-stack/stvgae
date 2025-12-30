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
                 alpha_ortho=1.0):

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
        self.alpha_addon = alpha_l1 * 10.0  # Addon 的惩罚通常比 Pathway 重一点
        self.alpha_ortho = alpha_ortho
        self.prune_threshold = prune_threshold

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Pruning (软剪枝)
            self.prune_weights(threshold=self.prune_threshold)

            self.logs['loss'].append(total_loss.item())
            self.logs['recon'].append(recon.item())

            if (epoch + 1) % log_interval == 0:
                pbar.write(
                    f"Ep {epoch + 1:03d} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Recon: {recon.item():.4f} | "
                    f"Ortho: {ortho.item():.4f}"
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







# # st_vgae/trainer.py
# import torch
# import torch.optim as optim
# from tqdm import tqdm
# from .loss import LossFunction
#
#
# class Trainer:
#     def __init__(self, model, dataset, device='cuda:0',
#                  lr=1e-3, weight_decay=1e-5,
#                  beta_start=0.0, beta_end=0.35, n_epochs_kl_warmup=200,
#                  alpha_l1=0.005,
#                  prune_threshold=1e-4,
#                  alpha_ortho=1.0):  # ✅ 修改：接收 alpha_ortho，移除 gamma
#
#         self.model = model.to(device)
#         self.device = device
#
#         self.x_input = dataset.x_norm.to(device)
#         self.x_raw = dataset.x_raw.to(device)
#         self.sf = dataset.scale_factor.to(device)
#         self.edge_index = dataset.edge_index.to(device)
#
#         self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#         # ✅ 修改：LossFunction 不再需要 gamma
#         self.criterion = LossFunction()
#
#         self.beta_start = beta_start
#         self.beta_end = beta_end
#         self.warmup_steps = n_epochs_kl_warmup
#         self.alpha_l1 = alpha_l1
#         self.alpha_addon = alpha_l1 * 10.0
#         self.prune_threshold = prune_threshold
#         self.alpha_ortho = alpha_ortho  # 保存正交权重
#
#         self.logs = {'loss': [], 'recon': []}
#
#     def train(self, n_epochs=300, log_interval=25):
#         self.model.train()
#         pbar = tqdm(range(n_epochs), desc="Training")
#
#         for epoch in pbar:
#             self.optimizer.zero_grad()
#             current_beta = min(self.beta_end,
#                                self.beta_start + (epoch / self.warmup_steps) * (self.beta_end - self.beta_start))
#
#             mean, disp, pi, mu, log_var, z = self.model(self.x_input, self.edge_index, self.sf)
#
#             # ✅ 修改：使用 alpha_ortho 且不再接收 spatial_loss
#             # 注意：criterion 的返回值变了 (loss, recon, kl, ortho)
#             total_loss, recon, kl, ortho = self.criterion(
#                 mean=mean, disp=disp, pi=pi, x=self.x_raw, mu=mu, log_var=log_var,
#                 current_beta=current_beta,
#                 decoder_layer=self.model.decoder,
#                 alpha_l1=self.alpha_l1,
#                 alpha_addon=self.alpha_addon,
#                 alpha_ortho=self.alpha_ortho
#             )
#
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
#             self.optimizer.step()
#
#             # Pruning
#             self.prune_weights(threshold=self.prune_threshold)
#
#             self.logs['loss'].append(total_loss.item())
#             self.logs['recon'].append(recon.item())
#
#             if (epoch + 1) % log_interval == 0:
#                 # 打印日志
#                 pbar.write(
#                     f"Ep {epoch + 1:03d} | "
#                     f"Loss: {total_loss.item():.4f} | "
#                     f"Recon: {recon.item():.4f} | "
#                     f"Ortho: {ortho.item():.4f} | "  # 监控正交损失
#                     f"KL: {kl.item():.4f}"
#                 )
#
#     def prune_weights(self, threshold):
#         with torch.no_grad():
#             w_path = self.model.decoder.weight_pathway
#             mask = self.model.decoder.mask
#             inverse_mask = (1.0 - mask).bool()
#             w_path.data[inverse_mask] = torch.where(
#                 torch.abs(w_path.data[inverse_mask]) < threshold,
#                 torch.tensor(0.0, device=self.device),
#                 w_path.data[inverse_mask]
#             )
#             if self.model.decoder.weight_addon is not None:
#                 w_addon = self.model.decoder.weight_addon
#                 w_addon.data = torch.where(
#                     torch.abs(w_addon.data) < threshold,
#                     torch.tensor(0.0, device=self.device),
#                     w_addon.data
#                 )
#
#     def get_latent(self):
#         self.model.eval()
#         with torch.no_grad():
#             output = self.model(self.x_input, self.edge_index, self.sf)
#             z = output[-1]
#         return z.cpu().numpy()