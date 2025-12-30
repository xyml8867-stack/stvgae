# config.py
CONFIG = {
    'dataset': {
        'data_dir': './data/DLPFC',
        'sample_id': '151673',
        'gmt_path': './data/reactome/ReactomePathways.gmt',
        'metadata_file': 'metadata.tsv',
        'label_col': 'layer_guess'
    },
    'preprocessing': {
        'min_genes': 500,
        'min_cells': 3,
        'min_pathways_per_gene': 5,
        'min_genes_per_pathway': 5,
        'n_top_genes': 3000,
        'n_neighbors': 16
    },
    'model': {
        'hidden_dim_1': 512,
        'hidden_dim_2': 256,
        'heads': 4,  # GAT 需要多头注意力
        'dropout': 0.1,
        'n_addon': 20  # 给足空间去吸噪
    },
    'training': {
        'seed': 42,
        'epochs': 300,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'log_interval': 25,
        'beta_start': 0.0,
        'beta_end': 0.35,
        'n_epochs_kl_warmup': 200,
        'alpha': 0.05,
        'prune_threshold': 1e-4,

        # ❌ 已删除 gamma (静态图参数)
        # ✅ 新增 alpha_ortho (动态图参数)
        'alpha_ortho': 1.0
    }
}
