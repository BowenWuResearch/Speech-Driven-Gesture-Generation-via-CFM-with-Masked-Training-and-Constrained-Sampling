import torch as th
import numpy as np
from scipy import interpolate
from .smplx_models.smplx.joints import joints as JOINTS
from .rotation_utils import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    euler_angles_to_matrix
)


def mirror_smplx_poses(poses: np.ndarray) -> np.ndarray:
    n_frames = len(poses)
    poses = poses.reshape(n_frames, -1, 3)
    poses = th.from_numpy(poses).float()
    poses_euler = matrix_to_euler_angles(
        axis_angle_to_matrix(poses), 
        'XYZ'
    )
    mirrored_euler = th.zeros_like(poses_euler)
    for idx, name in JOINTS.items():
        if 'left' in name:
            src_name = name.replace('left', 'right')
        elif 'right' in name:
            src_name = name.replace('right', 'left')
        else:
            src_name = name
        src_idx = list(JOINTS.keys())[
            [v for v in JOINTS.values()].index(src_name)
        ]
        mirrored_euler[:, idx, 0] = poses_euler[:, src_idx, 0]
        mirrored_euler[:, idx, 1] = -poses_euler[:, src_idx, 1]
        mirrored_euler[:, idx, 2] = -poses_euler[:, src_idx, 2]
    mirrored_poses = matrix_to_axis_angle(
        euler_angles_to_matrix(mirrored_euler, 'XYZ')
    ).numpy()
    mirrored_poses = mirrored_poses.reshape(n_frames, -1)
    return mirrored_poses


def resample_data(data, nframes_new, mode='linear'):
    nframes = data.shape[0]
    x = np.arange(0, nframes)/(nframes-1)
    xnew = np.arange(0, nframes_new)/(nframes_new-1)

    data_out = np.zeros((nframes_new, data.shape[1]))
    for jj in range(data.shape[1]):
        y = data[:,jj]
        f = interpolate.interp1d(
            x, 
            y, 
            bounds_error=False, 
            kind=mode, 
            fill_value='extrapolate'
        )
        data_out[:,jj] = f(xnew)
    
    return data_out