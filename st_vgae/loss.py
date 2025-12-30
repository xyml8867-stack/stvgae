# st_vgae/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, mean, disp, pi, x, mu, log_var, current_beta,
                decoder_layer,
                alpha_l1=0.005,
                alpha_addon=0.05,
                alpha_ortho=1.0):

        # 1. ZINB
        recon_loss = self.zinb_loss(x, mean, disp, pi)

        # 2. KL
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        # 3. Ortho (保持正交，解耦 Pathway 和 Addon)
        n_path = decoder_layer.n_pathways
        z_path = mu[:, :n_path]
        z_addon = mu[:, n_path:]

        ortho_loss = torch.tensor(0.0, device=mu.device)
        if z_addon.shape[1] > 0:
            z_p_norm = F.normalize(z_path, dim=1)
            z_a_norm = F.normalize(z_addon, dim=1)
            ortho_loss = torch.mean(torch.abs(torch.mm(z_p_norm.t(), z_a_norm)))

        # 4. L1 Regularization (软约束的关键)
        l1_path = torch.mean(torch.abs(decoder_layer.weight_pathway) * (1.0 - decoder_layer.mask))

        l1_addon = torch.tensor(0.0, device=mu.device)
        if decoder_layer.weight_addon is not None:
            l1_addon = torch.mean(torch.abs(decoder_layer.weight_addon))

        total_loss = recon_loss + \
                     (current_beta * kld_loss) + \
                     (alpha_ortho * ortho_loss) + \
                     (alpha_l1 * l1_path) + \
                     (alpha_addon * l1_addon)

        return total_loss, recon_loss, kld_loss, ortho_loss

    def zinb_loss(self, x, mean, disp, pi, eps=1e-10):
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_case = t1 + t2 - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        return torch.mean(torch.mean(result, dim=1))