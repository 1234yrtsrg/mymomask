import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader

from data.t2m_dataset import Text2BlendshapeDataset
from models.vq.model import LengthEstimator
from models.vq.vq_trainer import LengthEstTrainer
from options.train_option import TrainLenEstOptions
from utils.dataset_paths import resolve_split_file
from utils.fixseed import fixseed

os.environ["OMP_NUM_THREADS"] = "1"

CLIP_VERSION = 'ViT-B/32'


def save_options(opt):
    expr_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    os.makedirs(expr_dir, exist_ok=True)

    file_name = pjoin(expr_dir, 'opt.txt')
    with open(file_name, 'wt', encoding='utf-8') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for key, value in sorted(vars(opt).items()):
            opt_file.write(f'{key}: {value}\n')
        opt_file.write('-------------- End ----------------\n')


def load_and_freeze_clip(device):
    import clip

    clip_model, _ = clip.load(CLIP_VERSION, device='cpu', jit=False)
    if str(device) != 'cpu':
        clip.model.convert_weights(clip_model)

    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    return clip_model.to(device)


def encode_text(clip_model, raw_text, device):
    import clip

    text = clip.tokenize(raw_text, truncate=True).to(device)
    return clip_model.encode_text(text).float()


def build_dataloader(dataset, opt, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=opt.num_workers,
        pin_memory=opt.gpu_id != -1,
    )


def main():
    parser = TrainLenEstOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")
    if opt.gpu_id != -1 and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log/length_estimator', opt.dataset_name, opt.name)
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    save_options(opt)

    train_split_file = resolve_split_file(opt.data_root, 'train')
    val_split_file = resolve_split_file(opt.data_root, 'val', fallback_splits=['test'])

    # The estimator only needs captions and aligned motion lengths.
    opt.pad_to_max_length = False
    train_dataset = Text2BlendshapeDataset(opt, mean=None, std=None, split_file=train_split_file)
    val_dataset = Text2BlendshapeDataset(opt, mean=None, std=None, split_file=val_split_file)

    train_loader = build_dataloader(train_dataset, opt, shuffle=True, drop_last=True)
    val_loader = build_dataloader(val_dataset, opt, shuffle=False, drop_last=False)

    num_length_classes = opt.max_motion_length // opt.unit_length + 1
    estimator = LengthEstimator(512, num_length_classes)
    text_encoder = load_and_freeze_clip(opt.device)

    print(f'Using device: {opt.device}')
    print(f'Training length estimator with {num_length_classes} classes.')

    trainer = LengthEstTrainer(opt, estimator, text_encoder, encode_text)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
