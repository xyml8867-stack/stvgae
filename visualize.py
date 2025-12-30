import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.cluster import KMeans  # <--- 新增这行
# 引入配置，为了拿到 sample_id
from config import CONFIG


def main():
    # ==========================================
    # 1. 路径设置 (使用绝对路径，防止找不到文件)
    # ==========================================
    # 获取 visualize.py 脚本所在的文件夹绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    sample_id = CONFIG['dataset']['sample_id']
    file_path = os.path.join(results_dir, f"result_{sample_id}.h5ad")

    print(f"Loading result from: {file_path}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print("\n❌ Error: Result file not found!")
        print(f"Looking for: {file_path}")
        print("💡 Hint: Please run 'main.py' first. It will create the 'results' folder and save the .h5ad file.")
        sys.exit(1)

    # ==========================================
    # 2. 读取数据
    # ==========================================
    adata = sc.read_h5ad(file_path)
    print(f"Data loaded successfully.")
    print(f" - Spots: {adata.shape[0]}")
    print(f" - Genes: {adata.shape[1]}")
    print(f" - Latent Features (Pathways): {adata.obsm['ST_VGAE'].shape[1]}")

    # ==========================================
    # 3. 绘图配置
    # ==========================================
    # 设置 Scanpy 的绘图风格
    sc.set_figure_params(dpi=150, facecolor='white', frameon=False)
    plt.rcParams['font.family'] = 'sans-serif'  # 防止字体报错

    # ==========================================
    # 4. 绘图 A: 空间聚类 (Spatial Domains)
    # ==========================================
    print("\n[Plotting] 1. Spatial Clustering...")

    # 基于 ST_VGAE 特征计算邻居图
    sc.pp.neighbors(adata, use_rep='ST_VGAE', n_neighbors=15)

    # 计算 UMAP (用于降维可视化)
    sc.tl.umap(adata)


    # ==========================================
    # [修改] 使用 K-Means 强制指定 7 个簇 (DLPFC Layer 1-6 + WM)
    # ==========================================
    print("   -> Running K-Means (n_clusters=7)...")

    # 1. 提取潜在向量 (N_spots x N_features)
    latent_feat = adata.obsm['ST_VGAE']

    # 2. 执行 K-Means
    # n_clusters=7 是 DLPFC 的标准设定
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10).fit(latent_feat)

    # 3. 将结果存回 adata.obs (必须转成字符串，否则会被当成连续数值画图)
    adata.obs['pathway_cluster'] = kmeans.labels_.astype(str)


    # 画图并保存
    plt.figure(figsize=(8, 8))
    sc.pl.spatial(
        adata,
        color='pathway_cluster',
        title=f"Spatial Domains (Sample {sample_id})",
        spot_size=120,  # 如果点太大或太小，调整这个数值
        palette='tab20',  # 颜色盘
        show=False
    )
    save_path_cluster = os.path.join(results_dir, f"spatial_cluster_{sample_id}.png")
    plt.savefig(save_path_cluster, bbox_inches='tight', dpi=300)
    print(f"   -> Saved to: {save_path_cluster}")

    # ==========================================
    # 5. 绘图 B: UMAP 投影
    # ==========================================
    print("[Plotting] 2. UMAP Projection...")
    plt.figure(figsize=(6, 6))
    sc.pl.umap(
        adata,
        color='pathway_cluster',
        title="UMAP of Pathway Activity",
        show=False
    )
    save_path_umap = os.path.join(results_dir, f"umap_{sample_id}.png")
    plt.savefig(save_path_umap, bbox_inches='tight', dpi=300)
    print(f"   -> Saved to: {save_path_umap}")

    # ==========================================
    # 6. 绘图 C: 最活跃的通路 (Top Active Pathway)
    # ==========================================
    print("[Plotting] 3. Top Active Pathway...")

    # 提取潜在向量
    latent_z = adata.obsm['ST_VGAE']

    # 计算每一列（每个通路）的标准差，标准差越大说明在切片上差异越明显
    pathway_std = np.std(latent_z, axis=0)
    top_idx = np.argmax(pathway_std)

    # 将该列数值赋给 obs 以便画图
    col_name = f'Pathway_Idx_{top_idx}'
    adata.obs[col_name] = latent_z[:, top_idx]

    plt.figure(figsize=(8, 8))
    sc.pl.spatial(
        adata,
        color=col_name,
        cmap='magma',  # 热力图颜色
        title=f"Top Active Pathway (Index {top_idx})",
        spot_size=150,
        show=False
    )
    save_path_pathway = os.path.join(results_dir, f"top_pathway_{sample_id}.png")
    plt.savefig(save_path_pathway, bbox_inches='tight', dpi=300)
    print(f"   -> Saved to: {save_path_pathway}")

    print("\n✅ All visualizations generated successfully!")
    print(f"📂 Check your results folder: {results_dir}")


if __name__ == "__main__":
    main()