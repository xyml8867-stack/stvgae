# utils/pathway_recorder.py
import os
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_and_save_first_layer(mask, gene_list, pathway_list, save_dir, pathway_info_df=None):
    """
    分析第一层 (Gene -> Pathway) 的连接情况。
    1. 保存 CSV 数据表。
    2. [修改版] 绘制离散入度分布图 (显示 1, 2, 3... 具体每个数值的通路数量)。
    3. 绘制所有通路的入度条形图 (Bar Plot)。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 数据格式转换
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    n_genes, n_pathways = mask.shape
    print(f"[Analysis] 正在分析第一层连接: {n_genes} Genes -> {n_pathways} Pathways ...")

    # 2. 收集数据
    data_rows = []

    # 遍历每一列（每一个通路）
    for j in range(n_pathways):
        pathway_id = pathway_list[j]
        connected_gene_indices = np.where(mask[:, j] > 0)[0]
        count = len(connected_gene_indices)

        if count > 0:
            connected_genes = [gene_list[idx] for idx in connected_gene_indices]
            pathway_name = pathway_id
            if pathway_info_df is not None and pathway_id in pathway_info_df.index:
                real_name = pathway_info_df.loc[pathway_id, 'pathway_name']
                if len(real_name) > 50:
                    real_name = real_name[:47] + "..."
                pathway_name = f"{real_name} ({pathway_id})"

            row = [pathway_id, pathway_name, count] + connected_genes
            data_rows.append(row)

    # 3. 排序
    data_rows.sort(key=lambda x: x[2], reverse=True)

    # ================= 任务 A: 保存 CSV =================
    max_len = max(len(row) for row in data_rows)
    columns = ["Pathway_ID", "Pathway_Name", "Gene_Count"] + [f"Gene_{i + 1}" for i in range(max_len - 3)]

    df = pd.DataFrame(data_rows, columns=columns[:max_len])
    csv_path = os.path.join(save_dir, "Layer1_Pathway_Connections.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Analysis] CSV 数据表已保存: {csv_path}")

    # ================= 任务 B: [核心修改] 绘制离散数值分布图 =================
    # 我们只统计入度 > 0 的数据
    counts = [row[2] for row in data_rows]

    # 设置可视化的截止点，比如只看前 60 个数值（1~60）
    # 如果有通路连接了 500 个基因，它会作为一个孤立的点，为了不让图太长，我们聚焦在密集区
    limit_x = 60

    plt.figure(figsize=(14, 7))  # 画宽一点

    # discrete=True: 关键参数！确保 1 是 1，2 是 2，不合并
    # binrange=(1, limit_x): 限制 X 轴范围，只画 1 到 60
    sns.histplot(counts, discrete=True, binrange=(0.5, limit_x + 0.5), color="#4c72b0", edgecolor="black", alpha=0.8)

    plt.title(f"Distribution of Pathway In-Degrees (Zoomed: 1 to {limit_x})", fontsize=16)
    plt.xlabel("Number of Connected Genes (In-Degree)", fontsize=14)
    plt.ylabel("Count of Pathways", fontsize=14)

    # 强制设置 X 轴刻度为 1, 2, 3... 这样你看得最清楚
    plt.xticks(np.arange(1, limit_x + 1, 1), rotation=90, fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    hist_path = os.path.join(save_dir, "Layer1_InDegree_Discrete.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"[Analysis] 离散数值分布图已保存: {hist_path}")

    # ================= 任务 C: 全量分批绘制条形图 =================
    batch_size = 50
    total_pathways = len(data_rows)
    num_batches = math.ceil(total_pathways / batch_size)

    print(f"[Analysis] 正在分批绘制所有通路详情 ({num_batches} 张图)...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_pathways)

        batch_data = data_rows[start_idx:end_idx]

        names = [row[1] for row in batch_data]
        batch_counts = [row[2] for row in batch_data]

        plt.figure(figsize=(12, 10))
        sns.barplot(x=batch_counts, y=names, palette="viridis")

        plt.title(f"Pathway Gene Counts (Rank {start_idx + 1} - {end_idx})", fontsize=16)
        plt.xlabel("Number of Connected Genes", fontsize=14)
        plt.ylabel("Pathway Name", fontsize=10)
        plt.tight_layout()

        batch_plot_path = os.path.join(save_dir, f"Layer1_BarPlot_Part_{i + 1}.png")
        plt.savefig(batch_plot_path, dpi=300)
        plt.close()

    print(f"[Analysis] 分析完成。请查看 {save_dir} 文件夹。")