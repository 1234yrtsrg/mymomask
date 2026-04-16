import os
from os.path import join as pjoin

import numpy as np
import torch

from data.t2m_dataset import Text2BlendshapeDataset
from models.mask_transformer.transformer import ResidualTransformer
from models.mask_transformer.transformer_trainer import ResidualTransformerTrainer
from models.vq.model import RVQVAE
from options.train_option import TrainT2MOptions
from utils.dataset_paths import resolve_split_file
from utils.distributed import (
    barrier,
    build_dataloader,
    cleanup_distributed,
    is_main_process,
    launch_training,
    setup_distributed,
    wrap_model_for_distributed,
)
from utils.fixseed import fixseed
from utils.get_opt import get_opt


def load_vq_model(opt):
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    input_width = getattr(vq_opt, 'input_width', 61)
    vq_model = RVQVAE(
        vq_opt,
        input_width,
        vq_opt.nb_code,
        vq_opt.code_dim,
        vq_opt.output_emb_width,
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        vq_opt.vq_norm,
    )
    ckpt_path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'finest.tar')
    if not os.path.exists(ckpt_path):
        ckpt_path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'latest.tar')
    ckpt = torch.load(ckpt_path, map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    if is_main_process(opt):
        print(f'Loading VQ Model {opt.vq_name}')
    return vq_model.to(opt.device), vq_opt


def train_worker(rank, opt):
    setup_distributed(rank, opt)
    fixseed(opt.seed)

    try:
        torch.autograd.set_detect_anomaly(True)

        opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
        opt.model_dir = pjoin(opt.save_root, 'model')
        opt.log_dir = pjoin('./log/res/', opt.dataset_name, opt.name)

        if is_main_process(opt):
            os.makedirs(opt.model_dir, exist_ok=True)
            os.makedirs(opt.log_dir, exist_ok=True)
        barrier()

        vq_model, vq_opt = load_vq_model(opt)
        clip_version = 'ViT-B/32'

        opt.num_tokens = vq_opt.nb_code
        opt.num_quantizers = vq_opt.num_quantizers

        res_transformer = ResidualTransformer(
            code_dim=vq_opt.code_dim,
            cond_mode='text',
            latent_dim=opt.latent_dim,
            ff_size=opt.ff_size,
            num_layers=opt.n_layers,
            num_heads=opt.n_heads,
            dropout=opt.dropout,
            clip_dim=512,
            shared_codebook=vq_opt.shared_codebook,
            cond_drop_prob=opt.cond_drop_prob,
            share_weight=opt.share_weight,
            clip_version=clip_version,
            opt=opt,
        )

        if is_main_process(opt):
            pc_transformer = sum(param.numel() for param in res_transformer.parameters_wo_clip())
            print(res_transformer)
            print('Total parameters of all models: {:.2f}M'.format(pc_transformer / 1000_000))

        res_transformer = wrap_model_for_distributed(res_transformer, opt)

        mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
        std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

        train_split_file = resolve_split_file(opt.data_root, 'train')
        val_split_file = resolve_split_file(opt.data_root, 'val', fallback_splits=['test'])

        train_dataset = Text2BlendshapeDataset(opt, mean, std, train_split_file)
        val_dataset = Text2BlendshapeDataset(opt, mean, std, val_split_file)

        train_loader = build_dataloader(train_dataset, opt, shuffle=True, drop_last=True, pin_memory=True)
        val_loader = build_dataloader(val_dataset, opt, shuffle=False, drop_last=True, pin_memory=True)

        trainer = ResidualTransformerTrainer(opt, res_transformer, vq_model)
        trainer.train(train_loader, val_loader)
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    launch_training(opt, train_worker)
