import os
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator
from options.eval_option import EvalT2MOptions
from utils.dataset_paths import configure_dataset_paths
from utils.fixseed import fixseed
from utils.get_opt import get_opt

clip_version = 'ViT-B/32'


def resolve_checkpoint(model_dir, preferred_files):
    for filename in preferred_files:
        ckpt_path = pjoin(model_dir, filename)
        if os.path.exists(ckpt_path):
            return ckpt_path
    raise FileNotFoundError(f'No checkpoint found in {model_dir}. Tried: {preferred_files}')


def load_checkpoint_with_fallback(model_dir, preferred_files, map_location):
    load_errors = []
    for filename in preferred_files:
        ckpt_path = pjoin(model_dir, filename)
        if not os.path.exists(ckpt_path):
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location=map_location)
            return ckpt, ckpt_path
        except Exception as exc:
            load_errors.append(f'{ckpt_path}: {exc}')

    if load_errors:
        joined_errors = '\n'.join(load_errors)
        raise RuntimeError(
            f'Found checkpoint files in {model_dir}, but none could be loaded.\n{joined_errors}'
        )

    raise FileNotFoundError(f'No checkpoint found in {model_dir}. Tried: {preferred_files}')


def load_vq_model(vq_opt, preferred_files=None):
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
    if preferred_files is None:
        preferred_files = ['finest.tar', 'latest.tar']
    model_dir = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model')
    ckpt, ckpt_path = load_checkpoint_with_fallback(
        model_dir,
        preferred_files,
        map_location='cpu',
    )
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt


def load_trans_model(model_opt, device, preferred_files=None):
    t2m_transformer = MaskTransformer(
        code_dim=model_opt.code_dim,
        cond_mode='text',
        latent_dim=model_opt.latent_dim,
        ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers,
        num_heads=model_opt.n_heads,
        dropout=model_opt.dropout,
        clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version=clip_version,
        opt=model_opt,
    )
    if preferred_files is None:
        preferred_files = ['net_best_acc.tar', 'net_best_loss.tar', 'latest.tar']
    model_dir = pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model')
    ckpt, ckpt_path = load_checkpoint_with_fallback(
        model_dir,
        preferred_files,
        map_location='cpu',
    )
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {model_opt.name} from {os.path.basename(ckpt_path)} epoch {ckpt["ep"]}!')
    return t2m_transformer.to(device)


def load_res_model(res_opt, vq_opt, device, preferred_files=None):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode='text',
        latent_dim=res_opt.latent_dim,
        ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers,
        num_heads=res_opt.n_heads,
        dropout=res_opt.dropout,
        clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight,
        clip_version=clip_version,
        opt=res_opt,
    )
    if preferred_files is None:
        preferred_files = ['net_best_loss.tar', 'latest.tar']
    model_dir = pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model')
    ckpt, ckpt_path = load_checkpoint_with_fallback(
        model_dir,
        preferred_files,
        map_location=device,
    )
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from {os.path.basename(ckpt_path)} epoch {ckpt["ep"]}!')
    return res_transformer.to(device)


def load_len_estimator(opt):
    num_length_classes = opt.max_motion_length // opt.unit_length + 1
    model = LengthEstimator(512, num_length_classes)
    model_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model')
    ckpt, ckpt_path = load_checkpoint_with_fallback(
        model_dir,
        ['finest.tar', 'latest.tar'],
        map_location=opt.device,
    )
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from {os.path.basename(ckpt_path)} epoch {ckpt["epoch"]}!')
    return model.to(opt.device)


def parse_prompts(opt):
    prompt_list = []
    length_list = []
    estimate_length = False

    if opt.text_prompt != "":
        prompt_list.append(opt.text_prompt)
        if opt.motion_length == 0:
            estimate_length = True
        else:
            length_list.append(opt.motion_length)
    elif opt.text_path != "":
        with open(opt.text_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    estimate_length = True
                    length_list = []
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise ValueError('A text prompt or a text prompt file is required.')

    return prompt_list, length_list, estimate_length


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    result_dir = pjoin('./generation', opt.ext)
    output_dir = pjoin(result_dir, 'blendshape')
    os.makedirs(output_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt.device)

    assert res_opt.vq_name == model_opt.vq_name

    t2m_transformer = load_trans_model(model_opt, opt.device)
    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()

    model_opt = configure_dataset_paths(model_opt)

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    def inv_transform(data):
        return data * std + mean

    captions, length_list, estimate_length = parse_prompts(opt)

    if estimate_length:
        print("No output length provided, estimating sequence length from text.")
        length_estimator = load_len_estimator(model_opt)
        length_estimator.eval()
        text_embedding = t2m_transformer.encode_text(captions)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)
        token_lens = Categorical(probs).sample()
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(opt.device).long()

    frame_lengths = (token_lens * 4).detach().cpu().tolist()

    for repeat_idx in range(opt.repeat_times):
        print(f'--> Repeat {repeat_idx}')
        with torch.no_grad():
            mids = t2m_transformer.generate(
                captions,
                token_lens,
                timesteps=opt.time_steps,
                cond_scale=opt.cond_scale,
                temperature=opt.temperature,
                topk_filter_thres=opt.topkr,
                gsample=opt.gumbel_sample,
            )
            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
            pred_sequences = vq_model.forward_decoder(mids)
            pred_sequences = pred_sequences.detach().cpu().numpy()
            pred_sequences = inv_transform(pred_sequences)

        for sample_idx, (caption, seq_len, sequence) in enumerate(zip(captions, frame_lengths, pred_sequences)):
            sequence = sequence[:seq_len]
            save_name = f'sample{sample_idx:02d}_repeat{repeat_idx}_len{seq_len}.npy'
            save_path = pjoin(output_dir, save_name)
            np.save(save_path, sequence)
            print(f'Saved sample {sample_idx} ({caption}) to {save_path}')
