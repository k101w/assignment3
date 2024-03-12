import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase
import pdb

# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        near = self.min_depth
        far = self.max_depth
        z_vals = torch.linspace(near,far,self.n_pts_per_ray)
        z_vals = z_vals.to(ray_bundle.directions.device)
        # TODO (Q1.4): Sample points from z values
        # ray_bundle (N 3)
        #Z_val (m)
        # sample_points: (N,m,3)  N,1,3; 64,1
        #c=torch.einsum('bim,ni->bnm', a.unsqueeze(1),b.unsqueeze(-1))
        ray_d = ray_bundle.directions
       
        sample_points = ray_bundle.origins[0] + torch.einsum('bim,ni->bnm',ray_d.unsqueeze(1), z_vals.unsqueeze(-1))
        # pdb.set_trace()
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=(z_vals * torch.ones_like(sample_points[..., :1])).permute(0,2,1),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}