import os
from os.path import join as pjoin
import codecs as cs
import random

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def _resolve_min_motion_length(opt):
    return getattr(opt, 'min_motion_length', 16)


def _resolve_max_motion_length(opt):
    return getattr(opt, 'max_motion_length', 196)


def _resolve_motion_upper_bound(opt):
    max_motion_length = _resolve_max_motion_length(opt)
    return None if max_motion_length is None or max_motion_length <= 0 else max_motion_length


def _should_keep_motion(length, min_motion_length, max_motion_length):
    if length < min_motion_length:
        return False
    if max_motion_length is not None and length > max_motion_length:
        return False
    return True


def _slice_motion_segment(motion, f_tag, to_tag, fps=20):
    start_idx = int(f_tag * fps)
    end_idx = int(to_tag * fps)
    return motion[start_idx:end_idx]


def _normalize_motion(motion, mean, std):
    return (motion - mean) / std


def _load_split_ids(split_file):
    with cs.open(split_file, 'r') as file:
        return [line.strip() for line in file.readlines()]


def _load_dataset_ids(motion_dir, split_file=None):
    if split_file is not None:
        return _load_split_ids(split_file)

    sample_ids = []
    for file_name in sorted(os.listdir(motion_dir)):
        if file_name.endswith('.npy'):
            sample_ids.append(os.path.splitext(file_name)[0])
    return sample_ids


def _load_caption_only_text_entries(text_path):
    captions = []
    with cs.open(text_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if not line:
                continue
            line_split = line.split('#')
            if not line_split:
                continue
            caption = line_split[0].strip()
            if caption:
                captions.append(caption)
    return captions


class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.data = []
        self.lengths = []

        for name in tqdm(_load_split_ids(split_file)):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as exc:
                print(exc)

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        motion = _normalize_motion(motion, self.mean, self.std)
        return motion


class BlendshapeDataset(MotionDataset):
    pass


class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = _resolve_max_motion_length(opt)
        self.min_motion_length = _resolve_min_motion_length(opt)
        self.motion_upper_bound = _resolve_motion_upper_bound(opt)

        data_dict = {}
        new_name_list = []
        length_list = []

        for name in tqdm(_load_split_ids(split_file)):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if not _should_keep_motion(len(motion), self.min_motion_length, self.motion_upper_bound):
                    continue

                text_data = []
                has_full_sequence_caption = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as file:
                    for line in file.readlines():
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict = {'caption': caption, 'tokens': tokens}
                        if f_tag == 0.0 and to_tag == 0.0:
                            has_full_sequence_caption = True
                            text_data.append(text_dict)
                        else:
                            try:
                                sub_motion = _slice_motion_segment(motion, f_tag, to_tag)
                                if not _should_keep_motion(len(sub_motion), self.min_motion_length, self.motion_upper_bound):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {
                                    'motion': sub_motion,
                                    'length': len(sub_motion),
                                    'text': [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(sub_motion))
                            except Exception:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if has_full_sequence_caption:
                    data_dict[name] = {'motion': motion, 'length': len(motion), 'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception:
                pass

        if new_name_list:
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        else:
            name_list, length_list = [], []

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        coin2 = np.random.choice(['single', 'single', 'double']) if self.opt.unit_length < 10 else 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        else:
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        motion = _normalize_motion(motion, self.mean, self.std)

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))],
                axis=0,
            )
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = _resolve_max_motion_length(opt)
        self.min_motion_length = _resolve_min_motion_length(opt)
        self.motion_upper_bound = _resolve_motion_upper_bound(opt)

        data_dict = {}
        new_name_list = []
        length_list = []

        for name in tqdm(_load_split_ids(split_file)):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if not _should_keep_motion(len(motion), self.min_motion_length, self.motion_upper_bound):
                    continue

                text_data = []
                has_full_sequence_caption = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as file:
                    for line in file.readlines():
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict = {'caption': caption, 'tokens': tokens}
                        if f_tag == 0.0 and to_tag == 0.0:
                            has_full_sequence_caption = True
                            text_data.append(text_dict)
                        else:
                            try:
                                sub_motion = _slice_motion_segment(motion, f_tag, to_tag)
                                if not _should_keep_motion(len(sub_motion), self.min_motion_length, self.motion_upper_bound):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {
                                    'motion': sub_motion,
                                    'length': len(sub_motion),
                                    'text': [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(sub_motion))
                            except Exception:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if has_full_sequence_caption:
                    data_dict[name] = {'motion': motion, 'length': len(motion), 'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception:
                pass

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        text_data = random.choice(text_list)
        caption = text_data['caption']

        coin2 = np.random.choice(['single', 'single', 'double']) if self.opt.unit_length < 10 else 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        else:
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        motion = _normalize_motion(motion, self.mean, self.std)

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))],
                axis=0,
            )
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)


class Text2BlendshapeDataset(data.Dataset):
    def __init__(self, opt, mean=None, std=None, split_file=None):
        self.opt = opt
        self.pointer = 0
        self.mean = mean
        self.std = std
        self.max_motion_length = _resolve_motion_upper_bound(opt)
        self.min_motion_length = _resolve_min_motion_length(opt)
        self.motion_upper_bound = _resolve_motion_upper_bound(opt)
        self.pad_to_max_length = getattr(opt, 'pad_to_max_length', True)
        self.random_crop = getattr(opt, 'random_crop', True)

        data_dict = {}
        name_list = []
        length_list = []

        for name in tqdm(_load_dataset_ids(opt.motion_dir, split_file)):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if not _should_keep_motion(len(motion), self.min_motion_length, self.motion_upper_bound):
                    continue

                captions = _load_caption_only_text_entries(pjoin(opt.text_dir, name + '.txt'))
                if not captions:
                    continue

                data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'captions': captions,
                }
                name_list.append(name)
                length_list.append(len(motion))
            except Exception as exc:
                print(exc)

        if name_list:
            name_list, length_list = zip(*sorted(zip(name_list, length_list), key=lambda x: x[1]))
        else:
            name_list, length_list = [], []

        self.data_dict = data_dict
        self.name_list = name_list
        self.length_arr = np.array(length_list)

    def inv_transform(self, data):
        if self.mean is None or self.std is None:
            return data
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        caption = random.choice(data['captions'])
        motion = data['motion'].copy()
        m_length = data['length']

        if getattr(self.opt, 'unit_length', 0) > 0:
            if self.random_crop:
                coin2 = np.random.choice(['single', 'single', 'double']) if self.opt.unit_length < 10 else 'single'
                if coin2 == 'double':
                    m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
                else:
                    m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
                if m_length <= 0:
                    m_length = min(len(motion), self.opt.unit_length)
                start_idx = random.randint(0, len(motion) - m_length) if len(motion) > m_length else 0
                motion = motion[start_idx:start_idx + m_length]
            else:
                aligned_length = (m_length // self.opt.unit_length) * self.opt.unit_length
                if aligned_length > 0:
                    m_length = aligned_length
                    motion = motion[:m_length]

        if self.mean is not None and self.std is not None:
            motion = _normalize_motion(motion, self.mean, self.std)

        if self.max_motion_length is not None:
            if len(motion) > self.max_motion_length:
                motion = motion[:self.max_motion_length]
                m_length = min(m_length, self.max_motion_length)
            elif self.pad_to_max_length and m_length < self.max_motion_length:
                motion = np.concatenate(
                    [motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]), dtype=motion.dtype)],
                    axis=0,
                )

        motion = torch.from_numpy(np.asarray(motion, dtype=np.float32))
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert self.max_motion_length is None or length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
