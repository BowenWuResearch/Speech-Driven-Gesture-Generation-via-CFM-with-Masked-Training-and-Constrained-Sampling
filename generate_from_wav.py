from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch as th
from loguru import logger
from modules.reproduce_utils import seed_everything
from modules.dataset import (
    Stats,
    split_x,
    POSE_FPS,
    WINDOW_SIZE,
    SR
)
from modules.tensor_utils import normalize, denormalize
from modules.render_utils import render_smplx_sequences
from modules.fm import FlowMatchingModel
from modules.audio_feature_extraction import load_audio, preprocess_audio
import time

def parse_args():
    p = ArgumentParser()
    p.add_argument(
        '--audio-path',
        type=str,
        help='Path to audio file.',
        default='assets/audios/2_scott_0_6_6.wav',
        required=True
    )
    p.add_argument(
        '--chkpt-path',
        type=str,
        help='Path to audio-to-gesture model checkpoint.',
        default='assets/release/250320/chkpt.pth'
    )
    p.add_argument(
        '--stats-path',
        type=str,
        default='assets/release/250320/stats.pkl',
        help='Path to stats file.'
    )
    p.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Path to output directory.'
    )
    p.add_argument(
        '--overlap-size',
        type=int,
        default=32,
        help='Overlap size between consecutive windows.'
    )
    p.add_argument(
        '--hum-id',
        type=int,
        default=1 # scott
    )
    p.add_argument(
        '--n-timesteps',
        type=int,
        default=20,
        help='Number of denoising steps.'
    )
    p.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed'
    )
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    device = 'cuda' if th.cuda.is_available() else 'cpu'

    # Load data
    chkpt = th.load(args.chkpt_path, map_location='cpu')
    cfg = chkpt['cfg']
    mirror_aug = cfg.get('augmentation', False) and cfg['augmentation'].get('mirror_aug', False)
    time_aug = cfg.get('augmentation', False) and cfg['augmentation'].get('time_aug', False)
    stats = Stats(
        mirror_aug=mirror_aug,
        time_aug=time_aug,
        device=device,
        filepath=args.stats_path
    )

    # Load model
    model = FlowMatchingModel.from_chkpt(args.chkpt_path).to(device)
    model.eval()

    # Save path
    out_dirpath = Path(args.output_dir)
    out_dirpath.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio_path = Path(args.audio_path)
    audio = load_audio(args.audio_path, sr=SR)
    n_frames = int(audio.shape[0] / SR * POSE_FPS)
    audio_feats = preprocess_audio(audio, SR, POSE_FPS, n_frames) # (n_frames, C)
    n_frames = audio_feats.shape[0]
    logger.info(f'Audio features shape: {audio_feats.shape}')

    n_samples = 1 # Generate n samples
    audio_feats = th.from_numpy(audio_feats).float().to(device)
    audio_feats = normalize(
        audio_feats, stats.audio_feat_mu, stats.audio_feat_std
    )
    audio_feats = audio_feats[None].repeat(n_samples, 1, 1)
    hum_id = th.tensor([args.hum_id] * n_samples).to(device)

    # Iterative generation based on WINDOW_SIZE (128 ~ 4s)
    # overlap as parameter
    hop_size = WINDOW_SIZE - args.overlap_size
    with th.no_grad():
        out = model.sample(
            local_cond=audio_feats[:, :WINDOW_SIZE],
            hum_id=hum_id,
            n_timesteps=args.n_timesteps
        )
    generated = [out]
    st = hop_size
    seed = th.zeros(n_samples, WINDOW_SIZE, model.x_dim).to(device)
    seed_mask = th.ones_like(seed).to(device)
    seed_mask[:, :args.overlap_size] = 0    
    while st + WINDOW_SIZE < n_frames:
        seed[:, :args.overlap_size] = out[:, -args.overlap_size:]
        start_time = time.perf_counter()
        with th.no_grad():
            out = model.sample(
                local_cond=audio_feats[:, st:st+WINDOW_SIZE],
                hum_id=hum_id,
                seed=seed,
                seed_mask=seed_mask,
                n_timesteps=args.n_timesteps
            )
        print(f"Time taken for sampling: {time.perf_counter() - start_time:.2f} s")
        generated.append(out[:, args.overlap_size:])
        st += hop_size
    out = th.cat(generated, dim=1) # (n_samples, T, C)

    # Eliminate little jerky motion near the boundary
    out = th.nn.functional.interpolate(out.transpose(1, 2), scale_factor=1/2, mode='linear', align_corners=True)
    out = th.nn.functional.interpolate(out, scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)

    # Rescale
    out_xs = denormalize(out, stats.x_mu, stats.x_std) # (n_samples, T, D)
    out_faces, out_poses = split_x(out_xs)

    npz_path_list = []
    for i in range(n_samples):
        npz_path = str(out_dirpath / f'{audio_path.stem}_hum_{args.hum_id}_samples_{i}.npz')
        np.savez(
            npz_path,
            poses=out_poses[i].cpu().numpy(),
            trans=np.zeros((out_poses[i].shape[0], 3)),
            expressions=out_faces[i].cpu().numpy(),
            betas=np.zeros((300,)),
            mocap_frame_rate=POSE_FPS,
            model='SMPLX_NEUTRAL_2020',
            gender='neutral'
        )
        npz_path_list.append(npz_path)

    # Render video
    render_smplx_sequences(
        npz_path_list,
        output_dir=out_dirpath,
        audio_filepath=args.audio_path,
        output_name=f'{audio_path.stem}_hum_{args.hum_id}_vis',
        max_n_cols=n_samples,
        render_video_fps=POSE_FPS
    )
