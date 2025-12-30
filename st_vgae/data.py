# st_vgae/data.py
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import os
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from .dataset import STGraphDataset
from .constants import REGISTRY_KEYS


def load_and_process_data(data_path, sample_id, gmt_path,
                          min_genes=200, min_cells=3,
                          n_top_genes=3000,
                          n_neighbors=16,  # GAT 需要邻居
                          filter_params=None,
                          metadata_file=None, label_col=None):
    if filter_params is None:
        filter_params = {'min_genes_per_pathway': 5}

    print(f"[Data] Loading {sample_id}...")
    adata = sc.read_visium(path=f"{data_path}/{sample_id}",
                           count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    # 加载 Label
    if metadata_file:
        meta_path = os.path.join(data_path, sample_id, metadata_file)
        if os.path.exists(meta_path):
            print(f"[Metadata] Loading labels from {metadata_file}...")
            sep = '\t' if meta_path.endswith('.tsv') or meta_path.endswith('.txt') else ','
            df_meta = pd.read_csv(meta_path, sep=sep, index_col=0)
            if adata.obs_names[0].endswith('-1') and not df_meta.index[0].endswith('-1'):
                df_meta.index = df_meta.index + '-1'
            adata.obs = adata.obs.join(df_meta, how='left')

    # 1. 基础过滤
    print(f"[Preprocess] Basic Filtering...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # 2. 备份原始计数 (用于 ZINB Loss)
    adata.layers['counts'] = adata.X.copy()

    # 3. 归一化 & Log1p
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # === 4. 回归旧逻辑：先取交集，再选 HVG ===
    print(f"[Filter] Intersecting with Reactome genes first...")

    # 读取 GMT 获取所有通路基因
    reactome_genes = set()
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            reactome_genes.update(parts[2:])

    # 取交集
    intersect_genes = list(reactome_genes.intersection(set(adata.var_names)))
    adata = adata[:, intersect_genes].copy()
    print(f"   -> Retained {len(intersect_genes)} genes present in Reactome.")

    # 在交集的基础上选 HVG
    print(f"[Feature Selection] Selecting Top {n_top_genes} HVGs from intersected genes...")
    # 注意：如果交集基因少于 3000，就全都要
    target_n = min(n_top_genes, len(adata.var_names))
    sc.pp.highly_variable_genes(adata, n_top_genes=target_n, flavor='seurat', subset=True)

    final_genes = adata.var_names.tolist()
    final_genes.sort()
    adata = adata[:, final_genes].copy()
    print(f"[Feature Selection] Final Input Genes: {len(final_genes)}")

    # 5. 构建 Mask (这次是纯粹的 Mask，不会有全 0 列)
    print(f"[Alignment] Building Mask...")
    mask, pathway_names = build_mask_from_genes(
        gmt_path,
        final_genes,
        min_genes_per_pathway=filter_params['min_genes_per_pathway']
    )
    print(f"[Alignment] Mask Shape: {mask.shape}")

    # 6. 准备数据矩阵
    if sp.issparse(adata.X):
        x_norm = adata.X.toarray()
    else:
        x_norm = adata.X.copy()

    # 获取原始计数 (注意要对应切片后的基因)
    raw_counts_sub = adata.layers['counts']
    if sp.issparse(raw_counts_sub):
        x_raw = raw_counts_sub.toarray()
    else:
        x_raw = raw_counts_sub.copy()

    lib_size = x_raw.sum(1)
    size_factors = lib_size.astype(np.float32)

    # 7. 建图
    print(f"[Graph] Building Spatial Graph (k={n_neighbors})...")
    edge_index = build_spatial_graph(adata, n_neighbors=n_neighbors)

    dataset = STGraphDataset(
        x_norm=x_norm,
        x_raw=x_raw,
        adj=edge_index,
        size_factors=size_factors
    )

    adata_aligned = adata.copy()
    return dataset, mask, pathway_names, adata_aligned


def build_mask_from_genes(gmt_path, target_genes_list, min_genes_per_pathway=5):
    target_genes_set = set(target_genes_list)
    gene_to_idx = {g: i for i, g in enumerate(target_genes_list)}
    n_genes = len(target_genes_list)

    mask_list = []
    kept_pathways = []

    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            pname = parts[0]
            genes_in_pathway = set(parts[2:])
            common_genes = genes_in_pathway.intersection(target_genes_set)

            if len(common_genes) >= min_genes_per_pathway:
                p_vector = np.zeros(n_genes, dtype=np.float32)
                for g in common_genes:
                    p_vector[gene_to_idx[g]] = 1.0
                mask_list.append(p_vector)
                kept_pathways.append(pname)

    if not mask_list:
        return torch.zeros((1, n_genes)), ["Dummy"]

    return torch.FloatTensor(np.stack(mask_list)), kept_pathways


def build_spatial_graph(adata, n_neighbors=16):
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    n = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(coords)
    dists, idx = nbrs.kneighbors(coords)
    dists, idx = dists[:, 1:], idx[:, 1:]
    src = np.repeat(np.arange(n), n_neighbors)
    tgt = idx.reshape(-1)
    return torch.tensor([src, tgt], dtype=torch.long)