import torch
import numpy as np
from st_vgae.data import load_and_process_data
from st_vgae.model import StVGAE
from st_vgae.trainer import Trainer
from st_vgae.utils import setup_seed
from config import CONFIG
import os


def main():
    # 1. Setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    setup_seed(CONFIG['training']['seed'])

    print("=== st-VGAE Pipeline (Original Baseline) ===")

    # 2. Data
    data_cfg = CONFIG['dataset']
    prep_cfg = CONFIG['preprocessing']

    dataset, mask, pathway_names, adata_aligned = load_and_process_data(
        data_path=data_cfg['data_dir'],
        sample_id=data_cfg['sample_id'],
        gmt_path=data_cfg['gmt_path'],
        metadata_file=data_cfg.get('metadata_file'),
        label_col=data_cfg.get('label_col'),
        min_genes=prep_cfg['min_genes'],
        min_cells=prep_cfg['min_cells'],
        n_top_genes=prep_cfg['n_top_genes'],  # 3000
        n_neighbors=prep_cfg['n_neighbors'],
        filter_params={
            'min_pathways_per_gene': prep_cfg['min_pathways_per_gene'],
            'min_genes_per_pathway': prep_cfg['min_genes_per_pathway']
        }
    )

    # 3. Model
    model_cfg = CONFIG['model']
    model = StVGAE(
        adata_shape=dataset.x_norm.shape,
        mask=mask.to(device),
        n_addon=model_cfg['n_addon'],
        hidden_dim_1=model_cfg['hidden_dim_1'],
        hidden_dim_2=model_cfg['hidden_dim_2'],
        heads=model_cfg['heads'],
        dropout=model_cfg['dropout']
    )

    # 4. Train
    train_cfg = CONFIG['training']
    trainer = Trainer(
        model=model,
        dataset=dataset,
        device=device,
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
        beta_start=train_cfg['beta_start'],
        beta_end=train_cfg['beta_end'],
        n_epochs_kl_warmup=train_cfg['n_epochs_kl_warmup'],
        alpha_l1=train_cfg['alpha'],
        prune_threshold=train_cfg['prune_threshold'],
        alpha_ortho=train_cfg['alpha_ortho']
    )

    trainer.train(n_epochs=train_cfg['epochs'], log_interval=train_cfg['log_interval'])

    # 5. Extract Latent & Save (手动提取模式)
    print("\n[Result] Extracting latent space...")
    model.eval()
    with torch.no_grad():
        # 调用模型的前向传播，拿到最后一个返回值 z
        output = model(dataset.x_norm.to(device), dataset.edge_index.to(device), dataset.scale_factor.to(device))
        z = output[-1]
        latent_z = z.cpu().numpy()

    # 保存隐变量
    adata_aligned.obsm['ST_VGAE'] = latent_z

    # 保存结果
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"result_{data_cfg['sample_id']}.h5ad")

    # 保存 Pathway names (放在 uns 里最稳妥)
    if 'uns' not in dir(adata_aligned):
        adata_aligned.uns = {}
    adata_aligned.uns['pathway_names'] = pathway_names

    adata_aligned.write(save_path)
    print(f"[Result] Saved to {save_path}")


if __name__ == "__main__":
    main()