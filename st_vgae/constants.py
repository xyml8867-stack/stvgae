# st_vgae/constants.py

# 1. 注册键名 (防止拼写错误)
class REGISTRY_KEYS:
    X_KEY = "X"
    COUNTS_KEY = "counts"
    SPATIAL_KEY = "spatial"       # 空间坐标键名
    SIZE_FACTORS = "size_factors"
    PATHWAY_MASK = "pathway_mask"
    LATENT_KEY = "stVGAE_latent"

# 2. 数值稳定性
EPSILON = 1e-10
MIN_DISPERSION = 1e-4
MAX_DISPERSION = 1e4

# 3. 默认参数兜底
DEFAULTS = {
    "DROPOUT": 0.1,
    "HIDDEN_DIM": 256
}