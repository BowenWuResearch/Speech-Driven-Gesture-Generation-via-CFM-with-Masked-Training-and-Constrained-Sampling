import os
import queue
import subprocess
import threading
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import torch as th
import imageio
import smplx
import trimesh
import pyrender
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyvirtualdisplay import Display
from loguru import logger


def add_audio_to_video(silent_video_path, audio_path, output_video_path):
    command = [
        'ffmpeg',
        '-y',
        '-i', silent_video_path,
        '-i', audio_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-c:v', 'copy',
        '-shortest',
        output_video_path
    ]

    try:
        subprocess.run(command, check=True)
        logger.info(f'Video with audio generated successfully: {output_video_path}')
    except subprocess.CalledProcessError as e:
        logger.info(f'Error occurred: {e}')


def convert_img_to_mp4(input_pattern, output_filepath, framerate=30):
    command = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(output_filepath),
        '-y'
    ]

    try:
        subprocess.run(command, check=True)
        logger.info(f'Video conversion successful. Output filepath: {str(output_filepath)}')
    except subprocess.CalledProcessError as e:
        logger.info(f'Error during video conversion: {e}')


def deg_to_rad(degrees):
    return degrees * np.pi / 180


def create_pose_camera(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def create_pose_light(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def create_scene_with_mesh(vertices, faces, pose_camera, pose_light, floor_y_offset=None):
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=[220, 220, 220])
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True, poses=np.eye(4))
    scene = pyrender.Scene(bg_color=[255, 255, 255])
    scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    scene.add(camera, pose=pose_camera)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(light, pose=pose_light)
    return scene


def render_one_frame(renderer, frame_idx, vertices_list, faces, global_floor_offset=None):
    pose_camera = create_pose_camera(angle_deg=-5)
    pose_light = create_pose_light(angle_deg=-30)
    figs = []
    for vtx in vertices_list:
        scene = create_scene_with_mesh(vtx, faces, pose_camera, pose_light, global_floor_offset)
        fig, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        figs.append(fig)
    return figs


def render_frames_and_enqueue(
    fids, vertices_all_list, faces, render_width, render_height, 
    fig_queue, subprocess_index, global_floor_offset=None
):
    fig_resolution = (render_width // 2, render_height)
    renderer = pyrender.OffscreenRenderer(*fig_resolution)
    
    for idx, fid in tqdm(enumerate(fids),
                         total=len(fids),
                         desc=f'Rendering subprocess={subprocess_index}',
                         leave=False,
                         position=subprocess_index):
        figs = render_one_frame(renderer, fid, vertices_all_list[idx], faces, global_floor_offset)
        fig_queue.put((fid, figs))
    renderer.delete()


def write_images_from_queue(fig_queue, output_dir: Path, img_filetype, max_n_cols: int = 4):
    while True:
        e = fig_queue.get()
        if e is None:
            break
        fid, figs = e
        filename = output_dir / f'frame_{fid}.{img_filetype}'

        # Merge imgs
        while len(figs) % max_n_cols != 0:
            figs.append(np.zeros_like(figs[0]))
        figs = np.stack(figs) # (n, h, w, c)
        H, W, C = figs.shape[1:]
        figs = figs.reshape(-1, max_n_cols, H, W, C) # (n_rows, max_n_cols, h, w, c)
        merged_fig = np.vstack([np.hstack(row) for row in figs]) # (n_rows * h, max_n_cols * w, c)

        # Convert merged_fig to PIL image
        pil_image = Image.fromarray(merged_fig.astype('uint8'))

        # Draw text on the image
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()
        text_position = (0, 0)  # Position at the left-upper corner
        draw.text(text_position, f'fid={fid}', fill="black", font=font)

        # Convert PIL image back to numpy array
        merged_fig = np.array(pil_image)

        try:
            imageio.imwrite(str(filename), merged_fig)
        except Exception as ex:
            logger.error(f'Error writing image {filename}: {ex}')
            raise ex


def sub_process_process_frame(
    subprocess_index,
    render_video_width,
    render_video_height,
    render_tmp_img_filetype,
    fids,
    vertices_all_list,
    faces,
    output_dir: Path,
    max_n_cols,
    global_floor_offset=None
):
    begin_ts = time.time()

    fig_queue = queue.Queue()
    render_frames_and_enqueue(
        fids, vertices_all_list, faces, render_video_width, render_video_height, 
        fig_queue, subprocess_index, global_floor_offset
    )
    fig_queue.put(None)

    write_images_from_queue(fig_queue, output_dir, render_tmp_img_filetype, max_n_cols)


def distribute_frames(frames, render_video_fps, num_processes, vertices_all_list):
    sample_interval = max(1, int(30 // render_video_fps))
    subproc_frame_ids = [[] for _ in range(num_processes)]
    subproc_vertices = [[] for _ in range(num_processes)]
    sampled_frame_id = 0

    for i in range(frames):
        if i % sample_interval != 0:
            continue
        subprocess_index = sampled_frame_id % num_processes
        subproc_frame_ids[subprocess_index].append(sampled_frame_id)
        subproc_vertices[subprocess_index].append([vertices[i] for vertices in vertices_all_list])
        sampled_frame_id += 1

    return subproc_frame_ids, subproc_vertices


def generate_silent_video(
    output_filepath,
    render_video_fps,
    render_video_width,
    render_video_height,
    num_processes,
    render_tmp_img_filetype,
    frames,
    vertices_all_list,
    faces,
    output_dir: Path,
    max_n_cols
):
    # Calculate true global floor offset using all frames before distribution
    global_floor_offset = min(
        vtx[:, 1].min() for vertices_all in vertices_all_list 
        for vtx in vertices_all
    ) - 0.02  # Place floor slightly below lowest point
    
    subproc_frame_ids, subproc_vertices = distribute_frames(
        frames, render_video_fps, num_processes, vertices_all_list
    )

    logger.info(f'generate_silent_video NumProcesses={num_processes} time={time.time():.2f}')
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(
            sub_process_process_frame,
            [
                (subprocess_index,
                 render_video_width,
                 render_video_height,
                 render_tmp_img_filetype,
                 subproc_frame_ids[subprocess_index],
                 subproc_vertices[subprocess_index],
                 faces,
                 output_dir,
                 max_n_cols,
                 global_floor_offset)  # Pass the global offset to each subprocess
                for subprocess_index in range(num_processes)
            ]
        )

    convert_img_to_mp4(str(output_dir / f'frame_%d.{render_tmp_img_filetype}'), output_filepath, render_video_fps)
    filenames = output_dir.glob(f'*.{render_tmp_img_filetype}')
    for filename in filenames:
        os.remove(filename)


def get_vertices_all(model, data, num_betas=300, num_expression_coeffs=100):
    n = data['poses'].shape[0]
    # beta = th.from_numpy(data_np_body['betas']).float().unsqueeze(0).cuda()
    # beta = beta.repeat(n, 1)
    expression = th.from_numpy(data['expressions'][:n]).float().cuda()
    betas = th.zeros(n, num_betas).float().cuda()
    # expression = th.zeros(n, num_expression_coeffs).float().cuda()
    jaw_pose = th.from_numpy(data['poses'][:n, 66:69]).float().cuda()
    pose = th.from_numpy(data['poses'][:n]).float().cuda()
    transl = th.from_numpy(data['trans'][:n]).float().cuda()
    output = model(
        betas=betas,
        expression=expression,
        transl=transl,
        jaw_pose=jaw_pose,
        global_orient=pose[:,:3],
        body_pose=pose[:,3:21*3+3],
        left_hand_pose=pose[:,25*3:40*3],
        right_hand_pose=pose[:,40*3:55*3],
        leye_pose=pose[:, 69:72],
        reye_pose=pose[:, 72:75],
        return_verts=True
    )
    return output['vertices'].cpu().detach().numpy()


def render_smplx_sequences(
    npz_path_list: list[Path] = None,
    data_list: list[dict[np.ndarray]] = None,
    output_dir: Path = Path('.'),
    audio_filepath: Path = None,
    output_name: str = 'output',
    model_folder='./modules/smplx_models',
    num_betas=300,
    num_expression_coeffs=100,
    max_n_cols=1,
    debug=False,
    render_video_fps=30
):
    assert npz_path_list is not None or data_list is not None, \
        'Either npz_path_list or data_list must be provided'
    
    logger.info(f'Rendering to {output_dir / output_name}.mp4')

    smplx_model = smplx.create(
        model_folder,
        model_type='smplx',
        gender='NEUTRAL_2020',
        use_face_contour=False,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        ext='npz',
        use_pca=False
    ).cuda().eval()

    smplx_faces = np.load(
        f'{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz',
        allow_pickle=True
    )['f']
    
    data_list = data_list or [np.load(npz_path, allow_pickle=True) for npz_path in npz_path_list]

    vertices_all_list = [
        get_vertices_all(
            smplx_model,
            data,
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs
        )
        for data in data_list
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        seconds = 3
    else:
        seconds = min([vertices_all.shape[0] for vertices_all in vertices_all_list]) / render_video_fps

    display = Display(visible=0, size=(1920, 720))
    display.start()
    DISPLAY = os.environ['DISPLAY']
    os.environ['DISPLAY'] = display.new_display_var

    silent_video_filepath = output_dir / 'silent_video.mp4'
    generate_silent_video(
        silent_video_filepath,
        render_video_fps,
        1920,
        720,
        os.cpu_count() - 1,
        'bmp',
        int(seconds*render_video_fps),
        vertices_all_list,
        smplx_faces,
        output_dir,
        max_n_cols
    )

    display.stop()
    os.environ['DISPLAY'] = DISPLAY

    if audio_filepath:
        video_filepath = Path(output_dir) / f'{output_name}.mp4'
        add_audio_to_video(silent_video_filepath, audio_filepath, video_filepath)
        os.remove(str(silent_video_filepath))

    return video_filepath
