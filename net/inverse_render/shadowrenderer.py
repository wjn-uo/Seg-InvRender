import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
from net.utils import rend_util
from net.inverse_render.sg_render import render_with_BRDF_multiple_sampling

TINY_NUMBER = 1e-6


def compute_relative_smoothness_loss(values, values_jittor):
    base = torch.maximum(values, values_jittor).clip(min=1e-6)
    difference = torch.sum(((values - values_jittor) / base) ** 2, dim=-1, keepdim=True)  # [..., 1]

    return difference


@torch.no_grad()
def sample_ray_equally(rays_o, rays_d, nSample=128, near=0.03, far=2.0):
    sample_dist = 2.0 / nSample  # Assuming the region of interest is a unit sphere
    z_vals = torch.linspace(0.0, 1.0, nSample)
    near = torch.Tensor([near]).expand([rays_o.shape[0], 1])
    far = torch.Tensor([far]).expand([rays_o.shape[0], 1])
    z_vals = near + (far - near) * z_vals[None, :]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
    mid_z_vals = z_vals + dists * 0.5
    pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
    pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(rays_o.shape[0], nSample)
    inside_sphere = pts_norm < 1.0
    return pts, dists, inside_sphere


class ShadowRenderer:
    def __init__(self,
                 sdf_network,
                 color_network,
                 envmap_material_network,
                 ray_tracer
                 ):

        self.sdf_network = sdf_network
        self.color_network = color_network
        self.envmap_material_network = envmap_material_network
        self.ray_tracer = ray_tracer

    def render_core(self,
                    input, cato_en
                    ):
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].squeeze(0)
        cato_en = cato_en.squeeze(0)

        rays_d, rays_o = rend_util.get_camera_params(uv, pose, intrinsics)

        rays_d = rays_d.squeeze(0)
        rays_o = rays_o.expand(rays_d.shape)

        idr_rgb_values = torch.ones_like(rays_o).float().cuda()
        sg_rgb_values = torch.ones_like(rays_o).float().cuda()
        sg_diffuse_rgb_values = torch.ones_like(rays_o).float().cuda()
        indir_rgb_values = torch.ones_like(rays_o).float().cuda()
        diffuse_albedo_values = torch.ones_like(rays_o).float().cuda()
        sg_specular_rgb_values = torch.ones_like(rays_o).float().cuda()
        roughness_values = torch.ones_like(rays_o).float().cuda()
        random_xi_roughness = torch.ones_like(rays_o).float().cuda()
        random_xi_diffuse_albedo = torch.ones_like(rays_o).float().cuda()
        vis = torch.zeros(rays_o.shape[0], 3).float().cuda()
        normals = torch.ones_like(rays_o).float().cuda()
        features = torch.zeros(rays_o.shape[0], 256).float().cuda()

        object_mask = object_mask.bool()
        object_mask = object_mask.squeeze(-1)
        with torch.no_grad():
            curr_start_points, surf_mask_intersect, dists = self.ray_tracer(
                sdf=lambda x: self.sdf_network(x)[:, :1], cam_loc=rays_o,
                object_mask=object_mask,
                ray_directions=rays_d)  # n_rays*n_samples,

        surf_mask_intersect = surf_mask_intersect & object_mask
        surf_pts = curr_start_points[surf_mask_intersect]

        if (surf_pts.shape[0] != 0):
            view_dirs = -rays_d[surf_mask_intersect]
            feature_vectors = self.sdf_network(surf_pts)[:, 1:]
            features[surf_mask_intersect] = feature_vectors

            with torch.enable_grad():
                g = self.sdf_network.gradient(surf_pts)

            normal = g / (torch.norm(g, dim=-1, keepdim=True) + 1e-6)  # ----> camera
            view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

            idr_rgb = self.color_network(surf_pts, normal, view_dirs, feature_vectors)
            idr_rgb_values[surf_mask_intersect] = idr_rgb

            sg_envmap_material = self.envmap_material_network(surf_pts, feature_vectors, cato_en[surf_mask_intersect])
            albebo = sg_envmap_material['sg_diffuse_albedo']

            roughness = sg_envmap_material['sg_roughness']
            specular_reflectance = sg_envmap_material['sg_specular_reflectance']

            lgtSGs = sg_envmap_material['sg_lgtSGs']

            ret = render_with_BRDF_multiple_sampling(lgtSGs, specular_reflectance, roughness, albebo, normal,
                                                     view_dirs, surf_pts, self.sdf_network, self.color_network,
                                                     self.ray_tracer)

            sg_rgb_values[surf_mask_intersect] = ret['sg_rgb']
            indir_rgb_values[surf_mask_intersect] = ret['indir_rgb']
            vis[surf_mask_intersect] = ret['vis']

            diffuse_albedo_values[surf_mask_intersect] = albebo
            roughness_values[surf_mask_intersect] = roughness

            random_xi_diffuse_albedo[surf_mask_intersect] = sg_envmap_material['random_xi_diffuse_albedo']
            random_xi_roughness[surf_mask_intersect] = sg_envmap_material['random_xi_roughness']
            normals[surf_mask_intersect] = normal

            sg_diffuse_rgb_values[surf_mask_intersect] = ret['diffuse_rgb']
            sg_specular_rgb_values[surf_mask_intersect] = ret['specular_rgb']

        background_mask = ~surf_mask_intersect
        light_dir = rays_d[background_mask]  # [..., 3], original point (camera) ---> envmap sphere
        background_rgb = self.get_background_rgb(self.envmap_material_network.get_lgtSGs(), light_dir)  # [..., 3]
        sg_rgb_values[background_mask] = background_rgb

        if self.envmap_material_network.envmap is not None:
            bg_rgb_values = render_envmap(self.envmap_material_network.envmap, rays_d[background_mask])
            sg_rgb_values[background_mask] = bg_rgb_values

        return {
            'points': curr_start_points,
            'network_object_mask': surf_mask_intersect,
            'object_mask': object_mask,
            'idr_rgb': idr_rgb_values,
            'sg_rgb': sg_rgb_values,
            'indir_rgb': indir_rgb_values,
            'normals': normals,
            'sg_diffuse_rgb': sg_diffuse_rgb_values,
            'sg_specular_rgb': sg_specular_rgb_values,
            'diffuse_albedo': diffuse_albedo_values,
            'roughness': roughness_values,
            'vis_shadow': vis,
            'random_xi_diffuse_albedo': random_xi_diffuse_albedo,
            'random_xi_roughness': random_xi_roughness
        }

    def get_background_rgb(self, lgtSGs, viewdirs):
        viewdirs = viewdirs.to(lgtSGs.device)
        viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

        # [M, 7] ---> [..., M, 7]
        dots_sh = list(viewdirs.shape[:-2])
        M = lgtSGs.shape[0]
        lgtSGs = lgtSGs.view([1, ] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])

        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
        lgtSGMus = torch.abs(lgtSGs[..., -3:])
        # [..., M, 3]
        rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
        rgb = torch.sum(rgb, dim=-2)  # [..., 3]
        # rgb = torch.clamp(rgb, min=0.0, max=1.0)
        return rgb


def render_envmap(envmap, viewdirs):
    H, W = envmap.shape[:2]
    envmap = envmap.permute(2, 0, 1).unsqueeze(0)

    phi = torch.arccos(viewdirs[:, 2]).reshape(-1) - TINY_NUMBER
    theta = torch.atan2(viewdirs[:, 1], viewdirs[:, 0]).reshape(-1)

    # normalize to [-1, 1]
    query_y = (phi / np.pi) * 2 - 1
    query_x = - theta / np.pi
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)

    rgb = F.grid_sample(envmap, grid, align_corners=True)
    rgb = rgb.squeeze().permute(1, 0)
    return rgb
