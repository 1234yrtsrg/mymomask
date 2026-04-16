import os
from os.path import join as pjoin

import numpy as np
from torch.utils.data import DataLoader

from data.t2m_dataset import Text2BlendshapeDataset
from utils.dataset_paths import resolve_split_file
from utils.get_opt import get_opt

def get_dataset_motion_loader(opt_path, batch_size, fname, device):
    opt = get_opt(opt_path, device)
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    opt.random_crop = False
    opt.pad_to_max_length = True
    split_file = resolve_split_file(opt.data_root, fname, fallback_splits=['test', 'val', 'train'])
    dataset = Text2BlendshapeDataset(opt, mean, std, split_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=True)

    print('Blendshape Dataset Loading Completed!!!')
    return dataloader, dataset
