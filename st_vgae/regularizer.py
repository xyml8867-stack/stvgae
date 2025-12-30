# st_vgae/regularizer.py

import torch
import torch.nn as nn


class ProxL1:
    """
    复刻 ExpiMap 的 Proximal L1 Regularizer。
    专门用于处理 '软掩码'：即对 Mask 为 0 的位置施加 L1 惩罚，使其趋向于 0。
    """

    def __init__(self, weight_param, mask, alpha_l1=0.001, decay_rate=1.0):
        """
        :param weight_param: 需要正则化的权重张量 (Parameter对象), 通常是 decoder.weight_pathway
        :param mask: 原始掩码矩阵 (1=通路内, 0=通路外)。形状需与 weight_param 广播兼容。
        :param alpha_l1: L1 惩罚力度 (决定了稀疏性的强弱)。
        :param decay_rate: 学习率衰减因子 (可选，如果 optimizer 有 LR schedule，这里最好同步)。
        """
        self.weight = weight_param
        self.alpha_l1 = alpha_l1
        self.decay_rate = decay_rate

        # === 核心逻辑：掩码取反 ===
        # ExpiMap 逻辑：我们只惩罚那些 "本不该存在" 的连接。
        # 如果 mask 是 None，则惩罚所有权重 (全局稀疏)。
        # 如果 mask 存在，~mask (即 mask==0 的位置) 变为 True。
        if mask is not None:
            # 确保 mask 在 GPU/CPU 设备正确
            self.register_mask(mask)
        else:
            self.inverse_mask = None

    def register_mask(self, mask):
        # 将 mask 转为 bool 并取反
        # 1 (通路) -> False (不惩罚)
        # 0 (非通路) -> True (惩罚)
        self.inverse_mask = ~(mask.bool())

        # 确保 inverse_mask 和权重在同一个设备上
        if self.weight.device != mask.device:
            self.inverse_mask = self.inverse_mask.to(self.weight.device)

    @torch.no_grad()
    def step(self, lr=1.0):
        """
        执行近端操作 (Proximal Operator)。
        公式: prox(w) = sign(w) * max(|w| - lambda*lr, 0)
        注意：这里的 lr 实际上是 optimizer 的当前学习率，或者包含在 alpha_l1 中。
        """
        # 真正的阈值 = 基础力度 * (可选的外部学习率因子)
        threshold = self.alpha_l1 * lr * self.decay_rate

        if self.inverse_mask is None:
            # 如果没有 Mask，对所有权重做软阈值
            self.weight.data = self._soft_threshold(self.weight.data, threshold)
        else:
            # === ExpiMap 核心：只修改 Inverse Mask 为 True 的部分 ===
            # 1. 提取出需要惩罚的部分
            masked_params = self.weight.data[self.inverse_mask]

            # 2. 对这部分做软阈值 (Soft Thresholding)
            # 这步操作会把很小的值直接变成 0
            new_params = self._soft_threshold(masked_params, threshold)

            # 3. 填回原矩阵
            self.weight.data[self.inverse_mask] = new_params

    @staticmethod
    def _soft_threshold(x, threshold):
        """
        软阈值函数实现
        Result = sign(x) * (|x| - threshold)+
        """
        # torch.sign(x) * torch.clamp(abs(x) - threshold, min=0)
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)