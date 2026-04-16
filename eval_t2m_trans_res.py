import os
from os.path import basename, join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.t2m_dataset import Text2BlendshapeDataset
from gen_t2m import load_res_model, load_trans_model, load_vq_model, resolve_checkpoint
from options.eval_option import EvalT2MOptions
from utils.dataset_paths import resolve_split_file
from utils.fixseed import fixseed
from utils.get_opt import get_opt


def resolve_eval_checkpoints(model_dir, which_epoch):
    if which_epoch == 'all':
        return sorted(file for file in os.listdir(model_dir) if file.endswith('.tar'))

    requested = [which_epoch]
    if not which_epoch.endswith('.tar'):
        requested.insert(0, f'{which_epoch}.tar')
    return [basename(resolve_checkpoint(model_dir, requested))]


def masked_generation_metrics(pred_motion, gt_motion, lengths):
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


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    log_path = pjoin(out_dir, f'{opt.ext}.log')

    model_opt = get_opt(pjoin(root_dir, 'opt.txt'), device=opt.device)
    vq_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt'), device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt'), device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt.device)
    assert res_opt.vq_name == model_opt.vq_name

    mean = np.load(pjoin(vq_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(vq_opt.meta_dir, 'std.npy'))
    mean_t = torch.tensor(mean, device=opt.device, dtype=torch.float32).view(1, 1, -1)
    std_t = torch.tensor(std, device=opt.device, dtype=torch.float32).view(1, 1, -1)

    model_opt.random_crop = False
    model_opt.pad_to_max_length = True
    split_file = resolve_split_file(model_opt.data_root, 'test', fallback_splits=['val', 'train'])
    eval_dataset = Text2BlendshapeDataset(model_opt, mean, std, split_file)
    eval_loader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=4)

    checkpoint_files = resolve_eval_checkpoints(model_dir, opt.which_epoch)

    vq_model = vq_model.to(opt.device)
    res_model = res_model.to(opt.device)
    vq_model.eval()
    res_model.eval()

    with open(log_path, 'w', encoding='utf-8') as log_file:
        for checkpoint_name in checkpoint_files:
            print(f'Loading checkpoint {checkpoint_name}')
            t2m_transformer = load_trans_model(model_opt, opt.device, preferred_files=[checkpoint_name])
            t2m_transformer = t2m_transformer.to(opt.device)
            t2m_transformer.eval()

            maes = []
            mses = []

            with torch.no_grad():
                for captions, motion, m_length in eval_loader:
                    motion = motion.to(opt.device).float()
                    m_length = m_length.to(opt.device).long()
                    token_lens = torch.clamp(m_length // model_opt.unit_length, min=1)

                    mids = t2m_transformer.generate(
                        list(captions),
                        token_lens,
                        timesteps=opt.time_steps,
                        cond_scale=opt.cond_scale,
                        temperature=opt.temperature,
                        topk_filter_thres=opt.topkr,
                        gsample=opt.gumbel_sample,
                    )
                    mids = res_model.generate(mids, list(captions), token_lens, temperature=1, cond_scale=5)
                    pred_motion = vq_model.forward_decoder(mids)

                    pred_motion_raw = pred_motion * std_t + mean_t
                    motion_raw = motion * std_t + mean_t
                    mae, mse = masked_generation_metrics(pred_motion_raw, motion_raw, m_length.cpu())
                    maes.append(mae)
                    mses.append(mse)

            summary = (
                f'{checkpoint_name}\n'
                f'  raw_mae: {np.mean(maes):.6f}\n'
                f'  raw_mse: {np.mean(mses):.6f}\n'
            )
            print(summary)
            print(summary, file=log_file, flush=True)
