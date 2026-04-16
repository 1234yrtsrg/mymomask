import os
from os.path import basename, join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.t2m_dataset import Text2BlendshapeDataset
from gen_t2m import load_vq_model, resolve_checkpoint
from options.vq_option import arg_parse
from utils.dataset_paths import resolve_split_file
from utils.get_opt import get_opt


def resolve_eval_checkpoints(model_dir, which_epoch):
    if which_epoch == 'all':
        return sorted(file for file in os.listdir(model_dir) if file.endswith('.tar'))

    requested = [which_epoch]
    if not which_epoch.endswith('.tar'):
        requested.insert(0, f'{which_epoch}.tar')
    return [basename(resolve_checkpoint(model_dir, requested))]


def masked_reconstruction_metrics(pred_motion, gt_motion, lengths):
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0

    for pred_item, gt_item, seq_len in zip(pred_motion, gt_motion, lengths.tolist()):
        pred_item = pred_item[:seq_len]
        gt_item = gt_item[:seq_len]
        diff = pred_item - gt_item
        total_abs += diff.abs().sum().item()
        total_sq += diff.square().sum().item()
        total_count += diff.numel()

    mae = total_abs / max(total_count, 1)
    mse = total_sq / max(total_count, 1)
    return mae, mse


if __name__ == "__main__":
    args = arg_parse(False)
    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))

    out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    log_path = pjoin(out_dir, f'{args.ext}.log')

    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=args.device)

    mean = np.load(pjoin(vq_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(vq_opt.meta_dir, 'std.npy'))
    mean_t = torch.tensor(mean, device=args.device, dtype=torch.float32).view(1, 1, -1)
    std_t = torch.tensor(std, device=args.device, dtype=torch.float32).view(1, 1, -1)

    vq_opt.random_crop = False
    vq_opt.pad_to_max_length = True
    split_file = resolve_split_file(vq_opt.data_root, 'test', fallback_splits=['val', 'train'])
    eval_dataset = Text2BlendshapeDataset(vq_opt, mean, std, split_file)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    checkpoint_files = resolve_eval_checkpoints(model_dir, args.which_epoch)

    with open(log_path, 'w', encoding='utf-8') as log_file:
        for checkpoint_name in checkpoint_files:
            net, ep = load_vq_model(vq_opt, preferred_files=[checkpoint_name])
            net = net.to(args.device)
            net.eval()

            loss_recons = []
            loss_commit = []
            perplexities = []
            maes = []
            mses = []

            with torch.no_grad():
                for _, motion, m_length in eval_loader:
                    motion = motion.to(args.device).float()
                    m_length = m_length.to(args.device).long()

                    pred_motion, commit_loss, perplexity = net(motion)
                    pred_motion_raw = pred_motion * std_t + mean_t
                    motion_raw = motion * std_t + mean_t

                    mae, mse = masked_reconstruction_metrics(pred_motion_raw, motion_raw, m_length.cpu())
                    maes.append(mae)
                    mses.append(mse)
                    loss_recons.append(torch.nn.functional.l1_loss(pred_motion, motion).item())
                    loss_commit.append(commit_loss.item())
                    perplexities.append(perplexity.item())

            summary = (
                f'{checkpoint_name} epoch {ep}\n'
                f'  recon_l1: {np.mean(loss_recons):.6f}\n'
                f'  commit: {np.mean(loss_commit):.6f}\n'
                f'  perplexity: {np.mean(perplexities):.6f}\n'
                f'  raw_mae: {np.mean(maes):.6f}\n'
                f'  raw_mse: {np.mean(mses):.6f}\n'
            )
            print(summary)
            print(summary, file=log_file, flush=True)
