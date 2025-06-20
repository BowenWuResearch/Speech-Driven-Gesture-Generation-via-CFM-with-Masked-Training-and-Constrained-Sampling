import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import smplx
import torch as th
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from loguru import logger
import librosa
from modules.audio_feature_extraction import load_audio, preprocess_audio
from modules.augment_utils import (
    mirror_smplx_poses,
    resample_data
)


RAW_DATA_PATH = Path('/home/wu/MountDirs/irserver1/database/BEAT/beat_v2.0.0/beat_v2.0.0/beat_english_v2.0.0')
PROCESSED_DATA_PATH = Path('./data')
POSE_FPS = 30
SR = 16000
FACE_DIM = 100
POSE_DIM = 165 # in axis-angle
X_DIM = FACE_DIM + POSE_DIM
WINDOW_SIZE = 128
WINDOW_LENGTH = WINDOW_SIZE / POSE_FPS # 4.27 sec


sid2hid = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    9: 7,
    10: 8,
    11: 9,
    12: 10,
    13: 11,
    15: 12,
    16: 13,
    17: 14,
    18: 15,
    20: 16,
    21: 17,
    22: 18,
    23: 19,
    24: 20,
    25: 21,
    27: 22,
    28: 23,
    30: 24
}


postfixes = {
    'mirror_aug': '_mirror_aug',
    'time_aug': '_time_aug'
}


def get_data_filepath(split: str, mirror_aug: bool = False, time_aug: bool = False):
    filename = f'{split}_data'
    if mirror_aug:
        filename += postfixes['mirror_aug']
    if time_aug:
        filename += postfixes['time_aug']
    return PROCESSED_DATA_PATH / f'{filename}.pkl'


def get_stats_filepath(mirror_aug: bool = False, time_aug: bool = False):
    filename = 'stats'
    if mirror_aug:
        filename += postfixes['mirror_aug']
    if time_aug:
        filename += postfixes['time_aug']
    return PROCESSED_DATA_PATH / f'{filename}.pkl'


def concat_x(face, pose):
    """
    Args:
        face (Tensor): (..., FACE_DIM)
        pose (Tensor): (..., POSE_DIM)

    Returns:
        x (Tensor): (..., FACE_DIM + POSE_DIM)
    """
    # Check dimension matching
    assert face.shape[:-1] == pose.shape[:-1]
    if isinstance(face, Tensor):
        x = th.cat([face, pose], dim=-1)
    elif isinstance(face, np.ndarray):
        x = np.concatenate([face, pose], axis=-1)
    else:
        raise ValueError('face and pose should be either Tensor or np.ndarray')
    return x


def split_x(x):
    """
    Args:
        x (Tensor): (..., FACE_DIM + POSE_DIM)

    Returns:
        face (Tensor): (..., FACE_DIM)
        pose (Tensor): (..., POSE_DIM)
    """
    face = x[..., :FACE_DIM]
    pose = x[..., FACE_DIM:FACE_DIM + POSE_DIM]
    return face, pose


@dataclass
class BatchData:
    x: Tensor # [face, pose]
    audio: Tensor
    audio_feat: Tensor
    hum_id: Tensor


@dataclass
class SequenceSample:
    x: Tensor # [face, pose]
    audio: Tensor
    audio_feat: Tensor
    hum_id: Tensor


@dataclass
class Stats:
    x_mu: Tensor
    x_std: Tensor
    audio_feat_mu: Tensor
    audio_feat_std: Tensor

    def __init__(self, mirror_aug: bool = False, time_aug: bool = False, device='cpu', filepath: str = None):
        if filepath is None:
            filepath = get_stats_filepath(mirror_aug=mirror_aug, time_aug=time_aug)
        with open(str(filepath), 'rb') as f:
            stats = pickle.load(f)
        self.x_mu = th.from_numpy(stats['x_mu']).float().to(device)
        self.x_std = th.from_numpy(stats['x_std']).float().to(device)
        self.audio_feat_mu = th.from_numpy(stats['audio_feat_mu']).float().to(device)
        self.audio_feat_std = th.from_numpy(stats['audio_feat_std']).float().to(device)


def cache_data(
    split: str, 
    sort_by_length: bool = True,
    include_spk: list[int] = None
):
    # load split csv file
    split_df = pd.read_csv(
        RAW_DATA_PATH / f'train_test_split.csv', header=None
    )
    # logger.debug(split_df.head())

    split_df = split_df[split_df[1] == split]
    # logger.debug(split_df.head())
    
    # Only include speaker 2
    if include_spk is not None:
        new_df = pd.DataFrame()
        for spk in include_spk:
            new_df = pd.concat([new_df, split_df[split_df[0].str.startswith(f'{spk}_')]])
        split_df = new_df

    # print(np.unique(split_df[0].apply(lambda x: x.split('_')[0]), return_counts=True))
    print(split_df.head())
    # assert 0

    def process_row(row):
        seq_id = row[0]

        # Load pose sequence
        smplxflame_30_filepath = RAW_DATA_PATH / f'smplxflame_30/{seq_id}.npz'
        with np.load(smplxflame_30_filepath, mmap_mode='r') as data:
            betas = data['betas']
            poses = data['poses']
            faces = data['expressions']
            trans = data['trans']

        if len(poses) < WINDOW_SIZE:
            logger.warning(f'{seq_id} is too short. Skip.')
            return None

        # Load audio
        audio_filepath = RAW_DATA_PATH / f'wave16k/{seq_id}.wav'
        audio = load_audio(audio_filepath)

        # compute translation delta
        delta_trans = np.diff(
            trans, 
            axis=0,
            prepend=np.zeros_like(trans[0:1])
        )
        
        # Downsample to POSE_FPS
        faces = faces[::int(POSE_FPS / POSE_FPS)]
        poses = poses[::int(POSE_FPS / POSE_FPS)]
        delta_trans = delta_trans[::int(POSE_FPS / POSE_FPS)]

        # Extract audio features as pose_fps
        audio_feats = preprocess_audio(
            audio, SR, anim_fps=POSE_FPS, anim_length=len(poses)
        )

        # Shorten
        min_dur = int(min(
            len(audio) / SR,
            len(audio_feats) / POSE_FPS,
            len(faces) / POSE_FPS,
            len(poses) / POSE_FPS,
            len(delta_trans) / POSE_FPS
        ))
        audio = audio[:int(min_dur * SR)]
        audio_feats = audio_feats[:int(min_dur * POSE_FPS)]
        poses = poses[:int(min_dur * POSE_FPS)]
        faces = faces[:int(min_dur * POSE_FPS)]
        delta_trans = delta_trans[:int(min_dur * POSE_FPS)]

        # Concat x
        xs = concat_x(faces, poses)

        return (
            seq_id,
            audio.astype(np.float32),
            audio_feats.astype(np.float32),
            xs.astype(np.float32),
            betas.astype(np.float32)
        )

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        futures = [pool.submit(process_row, row) for _, row in split_df.iterrows()]
        results = list(tqdm(as_completed(futures), total=len(futures), desc=f'Caching {split} data'))

    # Process results
    seq_id_list = []
    audio_list = []
    audio_feats_list = []
    xs_list = []
    betas_list = []
    for future in results:
        result = future.result()
        if result is None:
            continue
        seq_id, audio, audio_feats, xs, betas = result
        seq_id_list.append(seq_id)
        audio_list.append(audio)
        audio_feats_list.append(audio_feats)
        xs_list.append(xs)
        betas_list.append(betas)
        # speaker_id_list.append(int(seq_id.split('_')[0]))

    if sort_by_length:
        # Sort by length of xs_list
        lengths = [len(x) for x in xs_list]
        idxs = np.argsort(lengths)
        audio_list = [audio_list[i] for i in idxs]
        audio_feats_list = [audio_feats_list[i] for i in idxs]
        xs_list = [xs_list[i] for i in idxs]
        seq_id_list = [seq_id_list[i] for i in idxs]

    return dict(
        seq_id=seq_id_list,
        audio=audio_list,
        audio_feat=audio_feats_list,
        x=xs_list,
        beta=betas_list
    )


class BEATCustomDataset:
    def __init__(
        self, 
        split: str,
        mirror_aug: bool = False,
        time_aug: bool = False,
        include_spk: list[int] = None
    ):
        data_filepath = get_data_filepath(
            split, 
            mirror_aug=mirror_aug if split == 'train' else False,
            time_aug=time_aug if split == 'train' else False
        )

        # Load data if exists
        if data_filepath.exists():
            logger.info(f'Loading {split} data from {data_filepath}')
            self.data = pickle.load(open(data_filepath, 'rb'))

        # Create data if not exists
        else:
            # Cache data
            logger.info(f'Caching {split} data to {data_filepath}')
            if not PROCESSED_DATA_PATH.exists():
                PROCESSED_DATA_PATH.mkdir()
            self.data = cache_data(split, include_spk=include_spk)

            if split == 'train':
                # Data augmentation
                if mirror_aug:
                    raise NotImplementedError('Mirror augmentation is not implemented')
                    # Mirror before calculating stats as change mu and std
                    for i in range(len(self.data['x'])):
                        # Mirror pose
                        xs = self.data['x'][i]
                        faces, poses = split_x(xs)
                        poses = mirror_smplx_poses(poses)
                        xs = concat_x(faces, poses)
                        # Mirror translation along z axis
                        delta_trans = self.data['delta_tran'][i] # (T, 3)
                        delta_trans = delta_trans * np.array([[1, 1, -1]])
                        # Append
                        self.data['seq_id'] += [self.data['seq_id'][i] + '_mirror']
                        self.data['audio'] += [self.data['audio'][i]]
                        self.data['audio_feat'] += [self.data['audio_feat'][i]]
                        self.data['x'] += [xs]
                        self.data['delta_tran'] += [delta_trans]
                        self.data['beta'] += [self.data['beta'][i]]

                # Compute mu and std on training data
                x_mu = np.concatenate(self.data['x'], axis=0).mean(axis=0)
                x_std = np.concatenate(self.data['x'], axis=0).std(axis=0) + 1e-8
                audio_feat_mu = np.concatenate(self.data['audio_feat'], axis=0).mean(axis=0)
                audio_feat_std = np.concatenate(self.data['audio_feat'], axis=0).std(axis=0) + 1e-8
                stats = dict(
                    x_mu=x_mu.astype(np.float32),
                    x_std=x_std.astype(np.float32),
                    audio_feat_mu=audio_feat_mu.astype(np.float32),
                    audio_feat_std=audio_feat_std.astype(np.float32),
                )
                stats_filepath = get_stats_filepath(
                    mirror_aug=mirror_aug,
                    time_aug=time_aug
                )
                with open(str(stats_filepath), 'wb') as f:
                    pickle.dump(stats, f)

                if time_aug:
                    raise NotImplementedError('Time augmentation is not implemented')
                    # Time strech after calculating stats, as does not change mu and std
                    logger.info('Time stretching...')
                    for i in range(len(self.data['x'])):
                        # Time strech for 0.9x and 1.1x
                        for ratio in [0.9, 1.1]:
                            xs = self.data['x'][i]
                            xs = resample_data(xs, int(len(xs) / ratio))
                            audio = librosa.effects.time_stretch(
                                self.data['audio'][i], rate=ratio
                            )
                            audio_feats = self.data['audio_feat'][i]
                            audio_feats = resample_data(audio_feats, int(len(audio_feats) / ratio)
                            )
                            # Append
                            self.data['seq_id'] += [self.data['seq_id'][i] + f'_time_{ratio}']
                            self.data['audio'] += [audio]
                            self.data['audio_feat'] += [audio_feats]
                            self.data['x'] += [xs]
                            self.data['beta'] += [self.data['beta'][i]]

            # Save data
            with open(data_filepath, 'wb') as f:
                pickle.dump(self.data, f)

        # Log total squence length
        logger.info(f'{split} total sequences: {len(self.data["x"])}')
        total_frames = sum([len(m) for m in self.data['x']])
        logger.info(f'{split} total frames: {total_frames:,}')

        # Total duration
        total_duration = total_frames / POSE_FPS / 3600
        logger.info(f'{split} total duration: {total_duration:.2f} hours')

        # Accumulate size
        self.accum_size = np.cumsum([
            len(m) - int(WINDOW_LENGTH * POSE_FPS)
            for m in self.data['x']
        ])

    def __len__(self):
        return self.accum_size[-1]

    def __getitem__(self, idx: int, win_len: float = None):
        # Find the index of the sequence
        seq_idx = np.searchsorted(self.accum_size, idx, side='right')
        if seq_idx > 0:
            idx = idx - self.accum_size[seq_idx - 1]

        # Get the sequence
        seq_id = self.data['seq_id'][seq_idx]
        spk_id = int(seq_id.split('_')[0])
        hum_id = sid2hid[spk_id]
        xs = self.data['x'][seq_idx]
        audio = self.data['audio'][seq_idx]
        audio_feats = self.data['audio_feat'][seq_idx]

        # Slice
        audio_idx = int(idx / POSE_FPS * SR)
        if win_len is not None:
            win_size = int(win_len * POSE_FPS)
            end_idx = idx + win_size
            audio_win_size = int(win_len * SR)
            audio_end_idx = audio_idx + audio_win_size
        else:
            end_idx = idx + WINDOW_SIZE
            audio_end_idx = audio_idx + int(WINDOW_LENGTH * SR)

        audio_slice = audio[audio_idx:audio_end_idx]
        audio_feats_slice = audio_feats[idx:end_idx]
        xs_slice = xs[idx:end_idx]
    
        return dict(
            audio=th.from_numpy(audio_slice).float(),
            audio_feat=th.from_numpy(audio_feats_slice).float(),
            x=th.from_numpy(xs_slice).float(),
            hum_id=th.tensor(hum_id).long()
        )

    def random_sample(self, win_len: float = None):
        idx = np.random.randint(self.accum_size[-1])
        return self.__getitem__(idx, win_len=win_len)
