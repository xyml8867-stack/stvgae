import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, latent_dim, heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim_1)
        self.norm1 = nn.LayerNorm(hidden_dim_1)
        self.gat1 = GATv2Conv(hidden_dim_1, hidden_dim_2, heads=heads, concat=False, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_dim_2, hidden_dim_2, heads=heads, concat=False, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim_2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_2, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, x, edge_index):
        h = self.linear1(x)
        h = self.norm1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.gat1(h, edge_index)
        h = self.activation(h)
        h = self.dropout(h)
        h_in = h
        h = self.gat2(h, edge_index)
        h = h + h_in
        h = self.activation(h)
        h = self.dropout(h)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class MaskedLinearDecoder(nn.Module):
    def __init__(self, n_pathways, n_addon, n_genes, mask):
        super(MaskedLinearDecoder, self).__init__()
        self.n_pathways = n_pathways
        self.n_addon = n_addon
        self.n_genes = n_genes

        # Pathway Decoder
        self.weight_pathway = nn.Parameter(torch.Tensor(n_genes, n_pathways))
        nn.init.xavier_normal_(self.weight_pathway)

        # 注册 Mask (注意：mask shape 是 [Pathways, Genes])
        # 转置后变成 [Genes, Pathways] 以匹配 weight_pathway
        self.register_buffer("mask", mask.t())

        # Addon Decoder
        if n_addon > 0:
            self.weight_addon = nn.Parameter(torch.Tensor(n_genes, n_addon))
            nn.init.xavier_normal_(self.weight_addon)
        else:
            self.register_parameter("weight_addon", None)

        full_dim = n_pathways + n_addon
        self.pi_decoder = nn.Linear(full_dim, n_genes)
        self.disp_decoder = nn.Linear(full_dim, n_genes)

    def forward(self, z):
        # 1. Pathway 重构
        z_p = z[:, :self.n_pathways]

        # ✅ 核心修复：硬掩码 (Hard Masking)
        # 在计算前，强制把 mask 为 0 的权重清零
        masked_weight = self.weight_pathway * self.mask

        out = torch.matmul(z_p, masked_weight.t())

        # 2. Addon 重构
        if self.n_addon > 0:
            z_a = z[:, self.n_pathways:]
            out_addon = torch.matmul(z_a, self.weight_addon.t())
            out = out + out_addon

        mean = F.softmax(out, dim=1)
        pi = torch.sigmoid(self.pi_decoder(z))
        disp = torch.exp(self.disp_decoder(z))
        return mean, disp, pi


class StVGAE(nn.Module):
    def __init__(self, adata_shape, mask, n_addon=32, hidden_dim_1=512, hidden_dim_2=256, heads=4, dropout=0.1):
        super(StVGAE, self).__init__()
        in_dim = adata_shape[1]
        n_pathways = mask.shape[0]
        total_dim = n_pathways + n_addon
        self.encoder = Encoder(in_dim, hidden_dim_1, hidden_dim_2, total_dim, heads, dropout)
        self.decoder = MaskedLinearDecoder(n_pathways, n_addon, in_dim, mask)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, edge_index, size_factors):
        mu, log_var = self.encoder(x, edge_index)
        z = self.reparameterize(mu, log_var)
        mean_prop, disp, pi = self.decoder(z)
        mean = mean_prop * size_factors
        return mean, disp, pi, mu, log_var, z