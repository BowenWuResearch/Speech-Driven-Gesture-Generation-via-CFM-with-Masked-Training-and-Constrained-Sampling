from pathlib import Path
import librosa
import soundfile as sf
import gradio as gr
import numpy as np
import torch as th
from loguru import logger
import os
import atexit
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
from modules.audio_feature_extraction import preprocess_audio

# Load model
device = 'cuda' if th.cuda.is_available() else 'cpu'
chkpt_path = 'chkpts/250320.pth'
stats_path = 'chkpts/stats.pkl'
stats = Stats(
    device=device,
    filepath=stats_path
)
model = FlowMatchingModel.from_chkpt(chkpt_path).to(device)
model.eval()

example_list = [
    ['assets/audios/2_scott_0_6_6.wav'],
    ['assets/audios/4_lawrence_0_6_6.wav'],
    ['assets/audios/25_goto_6_1_1.wav']
]

# Keep track of the last generated video
last_video_path = None
last_audio_path = None

def cleanup():
    global last_video_path
    global last_audio_path
    if last_video_path and os.path.exists(last_video_path):
        try:
            os.remove(last_video_path)
            logger.info(f"Cleaned up video file: {last_video_path}")
        except Exception as e:
            logger.error(f"Error cleaning up video file: {e}")
    if last_audio_path and os.path.exists(last_audio_path):
        try:
            os.remove(last_audio_path)
            logger.info(f"Cleaned up audio file: {last_audio_path}")
        except Exception as e:
            logger.error(f"Error cleaning up audio file: {e}")

def generate_from_audio(audio, seed, inpainting=False):
    print(seed, inpainting)
    
    global last_video_path
    global last_audio_path
    # Clean up previous video if it exists
    cleanup()
    
    seed_everything(seed)
    sr, audio = audio
    # Convert int16 (-32767~32767) to float (-1.0~1.0)
    audio = audio.astype(np.float32) / 32767.0
    # Resample to SR
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    print(sr)
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
    hum_id = th.tensor([1] * n_samples).to(device)
    # Iterative generation based on WINDOW_SIZE (128 ~ 4s)
    # overlap as parameter
    overlap_size = 32
    n_timesteps = 20
    hop_size = WINDOW_SIZE - overlap_size
    with th.no_grad():
        out = model.sample(
            local_cond=audio_feats[:, :WINDOW_SIZE],
            hum_id=hum_id,
            n_timesteps=n_timesteps
        )
    generated = [out]
    st = hop_size
    seed = th.zeros(n_samples, WINDOW_SIZE, model.x_dim).to(device)
    seed_mask = th.ones_like(seed).to(device)
    seed_mask[:, :overlap_size] = 0    
    while st + WINDOW_SIZE < n_frames:
        seed[:, :overlap_size] = out[:, -overlap_size:]
        with th.no_grad():
            out = model.sample(
                local_cond=audio_feats[:, st:st+WINDOW_SIZE],
                hum_id=hum_id,
                seed=seed,
                seed_mask=seed_mask,
                n_timesteps=n_timesteps,
                inpainting=inpainting
            )
        generated.append(out[:, overlap_size:])
        st += hop_size
    out = th.cat(generated, dim=1) # (n_samples, T, C)

    # Eliminate little jerky motion near the boundary
    out = th.nn.functional.interpolate(out.transpose(1, 2), scale_factor=1/2, mode='linear', align_corners=True)
    out = th.nn.functional.interpolate(out, scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)

    # Rescale
    out_xs = denormalize(out, stats.x_mu, stats.x_std) # (n_samples, T, D)
    out_faces, out_poses = split_x(out_xs)

    data_dict = {
        'poses': out_poses[0].cpu().numpy(),
        'trans': np.zeros((out_poses[0].shape[0], 3)),
        'expressions': out_faces[0].cpu().numpy(),
        'betas': np.zeros((300,)),
        'mocap_frame_rate': POSE_FPS,
        'model': 'SMPLX_NEUTRAL_2020',
        'gender': 'neutral'
    }

    # Render the generated sequence
    last_audio_path = 'temp.wav'
    sf.write(last_audio_path, audio, SR)
    video_path = render_smplx_sequences(
        data_list=[data_dict],
        output_dir=Path('.'),
        audio_filepath=Path(last_audio_path),
        output_name='temp',
        max_n_cols=1,
        render_video_fps=POSE_FPS
    )
    last_video_path = str(video_path)
    return last_video_path

with gr.Blocks() as demo:
    gr.Markdown('# Speech-driven Gesture Generator')
    with gr.Row(equal_height=True):
        with gr.Column():
            audio_input = gr.Audio(
                label='Input audio'
            )
            gr.Markdown('## Parameters')
            inpainting_input = gr.Checkbox(value=False, label='Use inpainting for transition (baseline)')
            seed_input = gr.Number(value=0, label='Seed (Change for gesture variations)')
            button = gr.Button('Generate')
        video_output = gr.Video(label='Generated Gesture')

    gr.Examples(
        examples=example_list,
        inputs=[audio_input],
        label="Example Audios"
    )
    
    button.click(
        fn=generate_from_audio,
        inputs=[audio_input, seed_input, inpainting_input],
        outputs=video_output
    )

if __name__ == '__main__':
    # Register cleanup function to run when the program exits
    atexit.register(cleanup)
    demo.launch(share=True)