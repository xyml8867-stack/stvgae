import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MaskedLinear(nn.Linear):
    """
    借鉴 expiMap: 带掩膜约束的线性层。
    用于 Decoder，将 Pathway 映射回 Genes。
    """

    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer('mask', mask)

    def forward(self, input):
        # 严谨操作：确保 mask 和 weight 在同一设备且类型一致
        return F.linear(input, self.weight * self.mask, self.bias)


class GATEncoderBlock(nn.Module):
    """
    借鉴 stClinic: GAT 编码块。
    封装了 GATConv + LayerNorm + Activation + Dropout
    """

    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0, concat=False, use_bn=False):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=heads, concat=concat, dropout=dropout)
        # stClinic 和 Word 文档建议使用 LayerNorm
        self.norm = nn.LayerNorm(out_dim) if not use_bn else nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.norm(x)
        x = F.elu(x)  # GAT 常用 ELU 激活
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """
    对应 Word 文档的 4 层结构，但参数化了维度。
    Layers: Linear -> GAT -> Linear -> (Mu/Var)
    """

    def __init__(self, in_dim, hidden_dim, latent_dim, heads=3, dropout=0.1):
        super().__init__()

        # Layer 1: Input Projection
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Layer 2: Spatial GAT (stClinic core)
        # 这里的 hidden_dim 对应文档中的 256
        self.gat_layer = GATEncoderBlock(512, hidden_dim, heads=heads, dropout=dropout, concat=False)

        # Layer 3: Deep Abstraction
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Layer 4: Latent Projection (Pathway Dimensions)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.var_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        h = self.fc1(x)
        h = self.gat_layer(h, edge_index)
        h = self.fc3(h)
        return self.mu_layer(h), self.var_layer(h)


class Decoder(nn.Module):
    """
    借鉴 expiMap: 线性可解释 ZINB Decoder。
    """

    def __init__(self, n_pathways, n_genes, mask, dropout=0.0):
        super().__init__()

        # 支路 A (Mean/Expr): 必须受 Mask 约束
        # Mask shape: [n_pathways, n_genes] -> transpose to [n_genes, n_pathways] for Linear
        self.expr_decoder = MaskedLinear(n_pathways, n_genes, mask.t())

        # 支路 B (Dispersion): 全连接
        self.disp_decoder = nn.Linear(n_pathways, n_genes)

        # 支路 C (Dropout/Pi): 全连接
        self.pi_decoder = nn.Linear(n_pathways, n_genes)

    def forward(self, z, size_factors):
        # 1. Expression (Mean)
        # expiMap 使用 softmax 来确保比例，然后乘以 library size
        _scale = self.expr_decoder(z)
        scale = F.softmax(_scale, dim=-1)
        mean = scale * size_factors

        # 2. Dispersion (必须 > 0)
        disp = torch.exp(self.disp_decoder(z))

        # 3. Dropout Probability (0-1)
        pi = torch.sigmoid(self.pi_decoder(z))

        return mean, disp, pi