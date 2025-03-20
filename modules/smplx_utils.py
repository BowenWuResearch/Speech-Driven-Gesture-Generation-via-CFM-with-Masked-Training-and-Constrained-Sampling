from dataclasses import dataclass
import smplx
import torch as th
from loguru import logger

@dataclass
class SMPLXHelperOutput:
    joints: th.Tensor

class SMPLXHelper:
    def __init__(
        self, 
        model_folder: str = './modules/smplx_models', 
        device='cpu'
    ):
        self.model_folder = model_folder
        self.smplx_model = smplx.create(
            model_folder, 
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False
        ).to(device)
        logger.info(f'Loading SMPL-X model from {model_folder}.')

    def poses_to_output(
        self,
        poses: th.Tensor,
        betas: th.Tensor = None,
        expression: th.Tensor = None,
        transl: th.Tensor = None
    ) -> SMPLXHelperOutput:
        '''
        Convert rotation to position representation.
        
        Args:
            poses (th.Tensor): the axis-angle signals. (B, T, C).
        '''
        assert poses.ndim == 3, 'The input tensor must be 3D.'
        assert poses.shape[-1] == 165, 'The last dimension of the input tensor must be 165.'
        device = poses.device
        B, T, _ = poses.shape
        if betas is None:
            betas = th.zeros(B, 300).float().to(device)
        if expression is None:
            expression = th.zeros(B, 100).float().to(device)
        if transl is None:
            transl = th.zeros(B, 3).float().to(device)
        
        joints = []
        for t in range(T):
            out = self.smplx_model(
                betas=betas,
                expression=expression,
                transl=transl,
                jaw_pose=poses[:, t, 66:69],
                global_orient=poses[:, t, :3],
                body_pose=poses[:, t, 3:3+21*3],
                left_hand_pose=poses[:, t, 25*3:40*3],
                right_hand_pose=poses[:, t, 40*3:55*3],
                leye_pose=poses[:, t, 69:72],
                reye_pose=poses[:, t, 72:75],
                return_verts=True
            )
            joints.append(out.joints.view(B, -1))
        
        return SMPLXHelperOutput(
            joints=th.stack(joints, dim=1)
        )
    