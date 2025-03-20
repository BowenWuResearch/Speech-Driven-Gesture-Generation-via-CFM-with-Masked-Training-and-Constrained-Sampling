from pathlib import Path
import numpy as np
import random
import torch as th
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from contextlib import nullcontext
from .dataset import POSE_DIM, FACE_DIM
from .model import LDA
from .smplx_models.smplx.joint_names import SMPLX_JOINT_NAMES, SMPL_JOINT_NAMES
from typing import Callable
# 1 indicates smpl joint in x data
joint_mask = th.zeros(len(SMPLX_JOINT_NAMES), 3) # (n_joints, 3)
for i, name in enumerate(SMPLX_JOINT_NAMES):
    if name in SMPL_JOINT_NAMES:
        joint_mask[i] = 1.0
joint_mask = joint_mask.view(-1) # (pose_dim,)
joint_mask = th.cat([th.zeros(FACE_DIM,), joint_mask], dim=-1) # (x_dim,)


# def create_mask(
#     size: tuple[int],
#     prob: float,
#     device: str = 'cpu'
# ) -> th.FloatTensor:
#     '''
#     Assumed size: (B,T,C).
#     '''
#     assert len(size) == 3, 'Size should be (B,T,C)'
#     B, T, C = size

#     # mask = th.zeros(size, dtype=th.float, device=device)
#     # mask[:, :50, 154:157] = 1.0

#     frame_mask = th.zeros(size, dtype=th.float, device=device)
#     ch_mask = th.zeros(size, dtype=th.float, device=device)

#     flag = np.random.rand(B,) < prob
#     mask_idxs = np.where(flag)[0]
#     mask_count = sum(flag)
#     if mask_count == 0:
#         return frame_mask

#     # Frame mask
#     mask_len = np.random.randint(1, T, size=(mask_count,), dtype=int)
#     st = np.random.randint(0, T-mask_len, dtype=int) # (B,)
#     et = st + mask_len # (B,)
#     for i, idx in enumerate(mask_idxs):
#         frame_mask[idx, st[i]:et[i]] = 1.0
    
#     # Channel mask
#     mask_len = np.random.randint(1, C, size=(mask_count,), dtype=int)
#     st = np.random.randint(0, C-mask_len, dtype=int) # (B,)
#     et = st + mask_len # (B,)
#     for i, idx in enumerate(mask_idxs):
#         ch_mask[idx, :, st[i]:et[i]] = 1.0
    
#     mask = th.zeros_like(frame_mask)
#     modes = np.random.randint(0, 2, size=(mask_count,))
#     for i, idx in enumerate(mask_idxs):
#         if modes[i] == 0:
#             mask[idx] = frame_mask[idx]
#         elif modes[i] == 1:
#             mask[idx] = ch_mask[idx]
#         else:
#             mask[idx] = th.logical_and(frame_mask[idx], ch_mask[idx]).float()

#     return 1 - mask

def create_1d_mask(length, max_length: int = None):
    if max_length is None:
        max_length = length
    mask = th.zeros((length,), dtype=th.float)
    mask_len = random.randint(1, max_length-1)
    st = random.randint(0, length-mask_len)
    mask[st:st+mask_len] = 1
    return mask

# Same mask for all
def create_mask(
    size: tuple[int],
    device: str = 'cpu'
) -> th.FloatTensor:
    '''
    Assumed size: (B,T,C).
    '''
    # assert len(size) == 3, 'Size should be (B,T,C)'
    B, T, C = size

    # mode = random.choice([0, 1, 2, 3], )
    mode = np.random.choice([0, 2, 4], p=[0.2, 0.2, 0.6])
    # print('mode:', mode)
    if mode == 0:
        mask = create_1d_mask(T)
        mask = mask.unsqueeze(1).repeat(1, C) # (T,C)
    elif mode == 1:
        # SMPL mask for smpl conditional generation
        mask = joint_mask[None].repeat(T, 1) # (t,c)
        t_mask = create_1d_mask(T).unsqueeze(1).repeat(1, C) # (t,c)
        mask = th.logical_and(mask, t_mask).float()
    elif mode == 2:
        # Unconditional generation
        mask = th.zeros(T, C)
    elif mode == 3:
        # Channel mask
        mask = create_1d_mask(C)
        mask = mask.unsqueeze(0).repeat(T, 1) # (T,C)
    elif mode == 4:
        # Logical and mask
        mask1 = create_1d_mask(T)
        mask2 = create_1d_mask(C)
        mask = th.logical_and(mask1.unsqueeze(1).repeat(1, C), mask2.unsqueeze(0).repeat(T, 1)).float()
    else:
        raise ValueError(f'Unsupported model: {mode}')
    
    return 1 - mask.unsqueeze(0).repeat(B, 1, 1).to(device)

class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        x_dim: int,
        local_cond_dim: int,
        vel_loss_w: float = 0.0,
        sigma_min: float = 1e-4,
        **estimator_kwargs
    ):
        super().__init__()
        self.x_dim = x_dim
        self.sigma_min = sigma_min
        self.do_mask = True 
        self.vel_loss_w = vel_loss_w
        self.do_vel_loss = False
        if vel_loss_w > 0.0:
            self.do_vel_loss = True

        self.estimator = LDA(
            pose_dim=x_dim,
            l_cond_dim=local_cond_dim,
            **estimator_kwargs
        )

    def phi(self, x, z, t):
        return (1 - (1 - self.sigma_min) * t) * z + t * x

    def compute_loss(
        self,
        x: Tensor,
        local_cond: Tensor,
        hum_id: Tensor,
        device: str,
        amp_enabled: bool,
        amp_dtype: th.dtype
    ):
        """
        Args:
            x (torch.Tensor): samples from p(x_1), shape (B, T, Dx)
            local_cond (torch.Tensor): local condition, shape (B, T, Dc)
            hum_id (torch.Tensor): hum_id, shape (B,)
            t (torch.Tensor): timestep, shape (B, T)
        """
        loss = 0.0
        terms = {}

        # Sample noise p(x_0)
        z = th.randn_like(x)
        
        # Vector field that generates the path as training target
        u = x - (1 - self.sigma_min) * z

        # Random flow step
        t = th.rand(x.size(0), 1, 1, device=x.device, dtype=x.dtype)

        # Point on the probability path as input to the model (add noise)
        y = self.phi(x, z, t)

        # Create mask: 1 for adding noise, 0 for clean signal
        if self.do_mask:
            mask = create_mask(x.size(), device=x.device)
        else:
            mask = th.ones_like(x)
        y = mask * y + (1 - mask) * x

        # Model forward
        with th.autocast(device, enabled=amp_enabled, dtype=amp_dtype) \
            if amp_enabled else nullcontext():
            u_hat = self.estimator(y, local_cond, hum_id, t.flatten())

        # Compute loss
        mse = (u_hat - u) ** 2 # (B,T,D)
        fm_loss = (mse * mask).sum() / mask.sum()
        loss += fm_loss
        terms['fm_loss'] = fm_loss

        if self.do_vel_loss:
            vel_mask = th.logical_or(mask[:, 1:], mask[:, :-1]).float() # (B,T-1,D)
            vel = th.diff(u, dim=1) # (B,T-1,D)
            vel_hat = th.diff(u_hat, dim=1)
            vel_loss = (vel_hat - vel) ** 2
            vel_loss = (vel_loss * vel_mask).sum() / vel_mask.sum()
            loss += self.vel_loss_w * vel_loss
            terms['vel_loss'] = vel_loss

        terms['loss'] = loss
        return terms

    def solve_euler(
        self,
        x: Tensor,
        local_cond: Tensor,
        hum_id: Tensor,
        t_span: Tensor,
        seed: Tensor = None,
        seed_mask: Tensor = None,
        inpainting: bool = False,
        progress: bool = False,
        constrain_func: Callable = None
    ):
        """
        Fixed euler solver for ODEs.

        Args:
            x (torch.Tensor): random noise
            local_cond (torch.Tensor): local condition, shape (B, T, Dc)
            hum_id: hum_id, shape (B,)
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            seed (torch.Tensor): seed for generating, shape (B, T, Dx)
            seed_mask (torch.Tensor): mask for seed, shape (B, T, Dx). 
                1 for missing data, 0 for clean data.
        """

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        if progress:
            iter = tqdm(range(1, len(t_span)), desc='Solving', leave=False, dynamic_ncols=True)
        else:
            iter = range(1, len(t_span))

        # I am storing this because I can later plot it
        # by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        # sol = []
        for step in iter:
            # Sigma for the denoising step
            t_in = t * th.ones(x.size(0), device=x.device)
            
            if seed is not None:
                if inpainting:
                    # Traditional inpainting method
                    seed_t = self.phi(seed, th.randn_like(seed), t)
                    x = seed_mask * x + (1.0 - seed_mask) * seed_t
                elif self.do_mask:
                    # Replace masked region with clean seed signal
                    x = seed_mask * x + (1.0 - seed_mask) * seed

            # Flow matching estimation
            with th.no_grad():
                dpsi_dt = self.estimator(x, local_cond, hum_id, t_in)

            # Euler step
            x = x + dt * dpsi_dt
            if constrain_func is not None:
                x = constrain_func(x)
            # sol.append(x.detach())

            # Update time
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        if seed is not None:
            # Replace seed region as seed,
            # instead of using the model's output
            x = seed_mask * x + (1.0 - seed_mask) * seed

        # return sol[-1]
        return x.detach()

    def sample(
        self,
        local_cond: Tensor,
        hum_id: Tensor,
        z: Tensor = None,
        n_timesteps: int = 25,
        seed: Tensor = None,
        seed_mask: Tensor = None,
        inpainting: bool = False,
        progress: bool = False,
        constrain_func: Callable = None
    ) -> Tensor:
        """Reverse diffusion sampling

        Args:
            batch_size (int): number of samples to generate
            local_cond (torch.Tensor): local condition, shape (B, T, Dc)
            hum_id (torch.Tensor): hum_id, shape (B,)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. 
                Defaults to 1.0.
            seed (torch.Tensor): seed for generating, shape (B, T, Dx), 
                where T must equal to num_frames
            seed_mask (torch.Tensor): mask for seed, shape (B, T). 
                0 for seed, 1 for missing data.

        Returns:
            sample: generated 
                shape: (B, T, D)
        """
        device = next(self.parameters()).device
        batch_size, num_frames, _ = local_cond.size()
        if z is None:
            z = th.randn(batch_size, num_frames, self.x_dim, device=device)
        else:
            assert z.size()[:2] == (batch_size, num_frames), 'z must have' + \
            ' same batch and num frames with local cond.'

        if seed is not None:
            assert seed.size(1) == num_frames, f'seed must have {num_frames} frames. Got {seed.size(1)}.'
            assert seed_mask is not None, 'seed_mask must be provided if seed is provided.'
            assert seed_mask.size()[:2] == seed.size()[:2], f'seed_mask must have same batch size and time step as seed.'
            assert seed.size() == seed_mask.size()
        
        t_span = th.linspace(0, 1, n_timesteps + 1, device=device)
        return self.solve_euler(
            z,
            local_cond,
            hum_id,
            t_span,
            seed=seed,
            seed_mask=seed_mask,
            inpainting=inpainting,
            progress=progress,
            constrain_func=constrain_func
        )

    @staticmethod
    def from_chkpt(chkpt_path: Path):
        chkpt = th.load(chkpt_path, weights_only=True, map_location='cpu')
        model_kwargs = chkpt['cfg']['model']

        # Create model and load state
        model = FlowMatchingModel(**model_kwargs)
        if 'avg_model_state' in chkpt:
            model.load_state_dict(chkpt['avg_model_state'])
        else:
            model.load_state_dict(chkpt['model_state'])

        return model