# st_vgae/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class STGraphDataset(Dataset):
    """
    统一的数据容器，负责存储 Tensor 并预计算 Scale Factors
    """
    def __init__(self, x_norm, x_raw, adj, size_factors=None):
        """
        x_norm: (N, G) 归一化后的表达矩阵 (Encoder Input)
        x_raw:  (N, G) 原始计数矩阵 (Decoder Target)
        adj:    (2, E) 边索引 (Edge Index)
        """
        # 1. 使用 from_numpy 避免不必要的内存拷贝 (更高效)
        # 注意：前提是输入必须是 numpy array，我们在 data.py 里已经保证了是 .toarray()
        self.x_norm = torch.from_numpy(x_norm).float()
        self.x_raw = torch.from_numpy(x_raw).float()

        # 2. 确保 adj 是 LongTensor (PyG 要求)
        if isinstance(adj, torch.Tensor):
            self.edge_index = adj.long()
        else:
            self.edge_index = torch.LongTensor(adj)

        # 3. 计算 Scaling Factor (用于 ZINB)
        if size_factors is not None:
            self.scale_factor = torch.from_numpy(size_factors).float().view(-1, 1)
        else:
            library_size = self.x_raw.sum(dim=1, keepdim=True)
            # 使用中位数作为分母，比硬编码 10000 更安全
            median_lib_size = library_size.median()
            self.scale_factor = library_size / (median_lib_size + 1e-6)

    def __len__(self):
        return self.x_norm.shape[0]

    def __getitem__(self, idx):
        # 这里的 key 主要是给 DataLoader 用的
        # 虽然我们的 Trainer 是全图训练直接取属性，但保留这个接口是个好习惯
        return {
            'x_norm': self.x_norm[idx],
            'x_raw': self.x_raw[idx],
            'sf': self.scale_factor[idx]
        }