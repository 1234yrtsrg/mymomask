import os
from os.path import join as pjoin


LEGACY_MOTION_DIR_DATASETS = {'blendshape', 't2m', 'kit'}


def default_data_root(dataset_name):
    return pjoin('.', 'dataset', dataset_name)


def default_motion_dir(dataset_name, data_root):
    data_dir = pjoin(data_root, 'data')
    motions_dir = pjoin(data_root, 'motions')

    if dataset_name in LEGACY_MOTION_DIR_DATASETS:
        return motions_dir

    if os.path.exists(data_dir):
        return data_dir
    if os.path.exists(motions_dir):
        return motions_dir

    # For custom datasets, default to the newer `data/` layout.
    return data_dir


def default_text_dir(dataset_name, data_root):
    return pjoin(data_root, 'texts')


def default_dataset_meta_dir(data_root):
    return data_root


def configure_dataset_paths(opt):
    default_root = default_data_root(opt.dataset_name)
    if not hasattr(opt, 'data_root') or opt.data_root in ['', './dataset/blendshape']:
        opt.data_root = default_root

    if not hasattr(opt, 'motion_dir') or opt.motion_dir == '':
        opt.motion_dir = default_motion_dir(opt.dataset_name, opt.data_root)

    if not hasattr(opt, 'text_dir') or opt.text_dir == '':
        opt.text_dir = default_text_dir(opt.dataset_name, opt.data_root)

    opt.dataset_meta_dir = default_dataset_meta_dir(opt.data_root)
    return opt


def resolve_split_file(data_root, split_name, fallback_splits=None):
    split_candidates = [split_name]
    if fallback_splits:
        split_candidates.extend(fallback_splits)

    candidate_paths = []
    for candidate_name in split_candidates:
        candidate_paths.extend(
            [
                pjoin(data_root, f'{candidate_name}.txt'),
                pjoin(data_root, 'split', f'{candidate_name}.txt'),
                pjoin(data_root, 'splits', f'{candidate_name}.txt'),
                pjoin(data_root, 'lists', f'{candidate_name}.txt'),
            ]
        )

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(f'No split file found. Tried: {candidate_paths}')
