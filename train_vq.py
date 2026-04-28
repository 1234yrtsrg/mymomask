import os
from os.path import join as pjoin

import numpy as np
import torch

from data.t2m_dataset import BlendshapeDataset
from models.vq.model import RVQVAE
from models.vq.vq_trainer import RVQTokenizerTrainer
from options.vq_option import arg_parse
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

os.environ["OMP_NUM_THREADS"] = "1"


def train_worker(rank, opt):
    setup_distributed(rank, opt)
    fixseed(opt.seed)

    try:
        torch.autograd.set_detect_anomaly(True)

        opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
        opt.model_dir = pjoin(opt.save_root, 'model')
        opt.meta_dir = pjoin(opt.save_root, 'meta')
        opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

        if is_main_process(opt):
            os.makedirs(opt.model_dir, exist_ok=True)
            os.makedirs(opt.meta_dir, exist_ok=True)
            os.makedirs(opt.log_dir, exist_ok=True)
        barrier()

        if is_main_process(opt):
            print(f"Using Device: {opt.device}")

        mean = np.load(pjoin(opt.dataset_meta_dir, 'Mean.npy'))
        std = np.load(pjoin(opt.dataset_meta_dir, 'Std.npy'))

        train_split_file = resolve_split_file(opt.data_root, 'train')
        val_split_file = resolve_split_file(opt.data_root, 'val', fallback_splits=['test'])

        net = RVQVAE(
            opt,
            opt.input_width,
            opt.nb_code,
            opt.code_dim,
            opt.output_emb_width,
            opt.down_t,
            opt.stride_t,
            opt.width,
            opt.depth,
            opt.dilation_growth_rate,
            opt.vq_act,
            opt.vq_norm,
        )

        if is_main_process(opt):
            pc_vq = sum(param.numel() for param in net.parameters())
            print(net)
            print('Total parameters of all models: {}M'.format(pc_vq / 1000_000))

        net = wrap_model_for_distributed(net, opt)
        trainer = RVQTokenizerTrainer(opt, vq_model=net)

        train_dataset = BlendshapeDataset(opt, mean, std, train_split_file)
        val_dataset = BlendshapeDataset(opt, mean, std, val_split_file)

        train_loader = build_dataloader(train_dataset, opt, shuffle=True, drop_last=True, pin_memory=True)
        val_loader = build_dataloader(val_dataset, opt, shuffle=False, drop_last=False, pin_memory=True)

        trainer.train(train_loader, val_loader)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    opt = arg_parse(True)
    launch_training(opt, train_worker)
