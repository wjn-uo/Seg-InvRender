import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
from net.inverse_render.sg_render import render_with_BRDF
from net.utils import rend_util


class Surfpts:
    def __init__(self,
                 sdf_network,
                 ray_tracer
                 ):
        self.sdf_network = sdf_network
        self.ray_tracer = ray_tracer

    def render_core(self,
                    input
                    ):
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].squeeze(0)

        rays_d, rays_o = rend_util.get_camera_params(uv, pose, intrinsics)

        rays_d = rays_d.squeeze(0)
        rays_o = rays_o.expand(rays_d.shape)

        object_mask = object_mask.bool()
        object_mask = object_mask.squeeze(-1)
        with torch.no_grad():
            curr_start_points, surf_mask_intersect, dists = self.ray_tracer(
                sdf=lambda x: self.sdf_network(x)[:, :1], cam_loc=rays_o,
                object_mask=object_mask,
                ray_directions=rays_d)  # n_rays*n_samples,

        true_mask = (object_mask > 0.5).squeeze()
        surf_pts = curr_start_points[true_mask]

        pts = torch.zeros_like(rays_o).float().cuda()
        pts[true_mask] = surf_pts
        return {
            'points': pts
        }
