import ast
import os
from os.path import join as pjoin

import numpy as np
import torch

from gen_t2m import load_res_model, load_trans_model, load_vq_model
from options.eval_option import EvalT2MOptions
from utils.fixseed import fixseed
from utils.get_opt import get_opt


def parse_edit_sections(section_specs, seq_len_tokens):
    if not section_specs:
        raise ValueError('At least one --mask_edit_section entry is required for editing.')

    edit_mask = torch.zeros((1, seq_len_tokens), dtype=torch.bool)
    for section_spec in section_specs:
        start_raw, end_raw = section_spec.split(',')
        start_val = ast.literal_eval(start_raw)
        end_val = ast.literal_eval(end_raw)

        if isinstance(start_val, float) or isinstance(end_val, float):
            start_idx = int(float(start_val) * seq_len_tokens)
            end_idx = int(float(end_val) * seq_len_tokens)
        else:
            start_idx = int(start_val) // 4
            end_idx = int(end_val) // 4

        start_idx = max(0, min(seq_len_tokens, start_idx))
        end_idx = max(start_idx, min(seq_len_tokens, end_idx))
        edit_mask[:, start_idx:end_idx] = True

    return edit_mask


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    if opt.source_motion == '':
        raise ValueError('`--source_motion` is required for edit_t2m.py.')
    if opt.text_prompt == '':
        raise ValueError('`--text_prompt` is required for edit_t2m.py.')

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    result_dir = pjoin('./editing', opt.ext)
    output_dir = pjoin(result_dir, 'blendshape')
    os.makedirs(output_dir, exist_ok=True)

    model_opt = get_opt(pjoin(root_dir, 'opt.txt'), device=opt.device)
    vq_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt'), device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    t2m_transformer = load_trans_model(model_opt, opt.device)

    res_model = None
    if opt.use_res_model:
        res_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt'), device=opt.device)
        res_model = load_res_model(res_opt, vq_opt, opt.device)
        assert res_opt.vq_name == model_opt.vq_name

    mean = np.load(pjoin(vq_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(vq_opt.meta_dir, 'std.npy'))

    def normalize_motion(data):
        return (data - mean) / std

    def inv_transform(data):
        return data * std + mean

    source_motion = np.load(opt.source_motion).astype(np.float32)
    max_motion_length = getattr(model_opt, 'max_motion_length', source_motion.shape[0])
    source_length = min(len(source_motion), max_motion_length)
    source_motion = source_motion[:source_length]

    source_motion_norm = normalize_motion(source_motion)
    if source_length < max_motion_length:
        source_motion_norm = np.concatenate(
            [
                source_motion_norm,
                np.zeros((max_motion_length - source_length, source_motion_norm.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )

    source_motion_tensor = torch.from_numpy(source_motion_norm).unsqueeze(0).to(opt.device)
    token_lens = torch.clamp(torch.LongTensor([source_length // model_opt.unit_length]), min=1).to(opt.device)
    captions = [opt.text_prompt]

    vq_model = vq_model.to(opt.device).eval()
    t2m_transformer = t2m_transformer.to(opt.device).eval()
    if res_model is not None:
        res_model = res_model.to(opt.device).eval()

    with torch.no_grad():
        tokens, _ = vq_model.encode(source_motion_tensor)
        edit_mask = parse_edit_sections(opt.mask_edit_section, tokens.shape[1]).to(opt.device)

        mids = t2m_transformer.edit(
            captions,
            tokens[..., 0].clone(),
            token_lens,
            timesteps=opt.time_steps,
            cond_scale=opt.cond_scale,
            temperature=opt.temperature,
            topk_filter_thres=opt.topkr,
            gsample=opt.gumbel_sample,
            force_mask=opt.force_mask,
            edit_mask=edit_mask,
        )

        if res_model is not None:
            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        else:
            mids = mids.unsqueeze(-1)

        pred_motion = vq_model.forward_decoder(mids).detach().cpu().numpy()[0]

    pred_motion = inv_transform(pred_motion)[:source_length]
    source_motion = inv_transform(source_motion_tensor.detach().cpu().numpy()[0])[:source_length]

    pred_path = pjoin(output_dir, 'edited_sample.npy')
    source_path = pjoin(output_dir, 'source_sample.npy')
    np.save(pred_path, pred_motion)
    np.save(source_path, source_motion)

    print(f'Saved edited blendshape sequence to {pred_path}')
    print(f'Saved source blendshape sequence to {source_path}')
