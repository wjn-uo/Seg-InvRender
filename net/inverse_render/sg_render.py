import torch
import torch.nn.functional as F
import numpy as np

TINY_NUMBER = 1e-6


def render_envmap_sg(lgtSGs, viewdirs):
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


def get_light_rgbs(lgtSGs=None, incident_light_directions=None, light_kind='sg', device='cuda'):
    '''
    - args:
        - incident_light_directions: [sample_number, 3]
    - return:
        - light_rgbs: [light_num, sample_number, 3]
    '''
    if light_kind == 'sg':
        light_rgbs = render_envmap_sg(lgtSGs, incident_light_directions).reshape(-1, 3)  # [sample_number, 3]
    else:
        pass
    return light_rgbs


@torch.no_grad()
def compute_secondary_shading_effects(
        sdf_network,
        color_network,
        ray_tracer,
        surface_pts,
        surf2light,
        chunk_size=15000
):
    '''compute visibility for each point at each direction without visbility network
    - args:
        - tensoIR: tensoIR model is used to compute the visibility and indirect lighting
        - surface_pts: [N, 3] surface points location
        - surf2light: [N, 3], light incident direction for each surface point, pointing from surface to light
        - light_idx: [N, 1], index of lighitng
        - nSample: number of samples for each ray along incident light direction
    - return:
        - visibility_compute: [N, 1] visibility result by choosing some directions and then computing the density
        - indirect_light: [N, 3] indirect light in the corresponding direction
    '''

    visibility_compute = torch.zeros((surface_pts.shape[0]), dtype=torch.float32)  # [N, 1]
    indirect_light = torch.zeros((surface_pts.shape[0], 3), dtype=torch.float32)  # [N, 1]

    chunk_idxs_vis_compute = torch.split(torch.arange(surface_pts.shape[0]), chunk_size)
    for chunk_idx in chunk_idxs_vis_compute:
        chunk_surface_pts = surface_pts[chunk_idx]  # [chunk_size, 3]
        chunk_surf2light = surf2light[chunk_idx]  # [chunk_size, 3]
        vis_chunk, indirect_light_chunk = compute_vis_radiance(sdf_network,
                                                               color_network,
                                                               ray_tracer=ray_tracer,
                                                               surf_pts=chunk_surface_pts,
                                                               light_in_dir=chunk_surf2light)

        visibility_compute[chunk_idx] = vis_chunk.float()
        indirect_light[chunk_idx] = indirect_light_chunk

    visibility_compute = visibility_compute.reshape(-1, 1)  # [N, 1]
    indirect_light = indirect_light.reshape(-1, 3)  # [N, 3]

    return visibility_compute, indirect_light


@torch.no_grad()
def compute_vis_radiance(sdf_network, color_network, ray_tracer, surf_pts, light_in_dir):
    """

    :param sdf_network:
    :param color_network:
    :param ray_tracer:
    :param surf_pts:
    :param light_in_dir: point to light
    :return:
    """
    indirect_color = torch.zeros_like(surf_pts)

    object_mask = torch.ones(surf_pts.shape[0]).bool()
    with torch.no_grad():
        sec_points, sec_net_object_mask, sec_dist = ray_tracer(
            sdf=lambda x: sdf_network(x)[:, :1], cam_loc=surf_pts,
            object_mask=object_mask,
            ray_directions=light_in_dir)  # n_rays*n_samples,

    hit_points = sec_points[sec_net_object_mask]
    hit_viewdirs = -light_in_dir[sec_net_object_mask]

    if (hit_points.shape[0] != 0):
        feature_vectors = sdf_network(hit_points)[:, 1:]

        with torch.enable_grad():
            g = sdf_network.gradient(hit_points)

        normal = g / (torch.norm(g, dim=-1, keepdim=True) + 1e-6)  # ----> camera
        hit_viewdirs = hit_viewdirs / (torch.norm(hit_viewdirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

        indirect_color[sec_net_object_mask] = color_network(hit_points, normal, hit_viewdirs, feature_vectors)

    viss = (~sec_net_object_mask).float()
    return viss, indirect_color


def prepend_dims(tensor, shape):
    '''
    :param tensor: tensor of shape [a1, a2, ..., an]
    :param shape: shape to prepend, e.g., [b1, b2, ..., bm]
    :return: tensor of shape [b1, b2, ..., bm, a1, a2, ..., an]
    '''
    orig_shape = list(tensor.shape)
    tensor = tensor.view([1] * len(shape) + orig_shape).expand(shape + [-1] * len(orig_shape))
    return tensor


def sg_fn(upsilon, xi, lamb, mu):
    """
    spherical gaussian (SG) function
    :param upsilon: [..., 3]; input variable
    :param xi: [..., 3]
    :param lamb: [..., 1]
    :param mu: [..., 3]
    """

    return mu * torch.exp(lamb * (torch.sum(upsilon * xi, dim=-1, keepdim=True) - 1))


def rotate_to_normal(xyz, n):
    """
    rotate coordinates from local space to world space
    :param xyz: [..., 3], coordinates in local space which normal is z
    :param n: [..., 3], normal
    :return: vec: [..., 3]
    """

    x_axis = torch.zeros_like(n)
    x_axis[..., 0] = 1

    y_axis = torch.zeros_like(n)
    y_axis[..., 1] = 1

    vup = torch.where((n[..., 0:1] > 0.9).expand(n.shape), y_axis, x_axis)
    t = torch.cross(vup, n, dim=-1)  # [..., 3]
    t = t / (torch.norm(t, dim=-1, keepdim=True) + TINY_NUMBER)
    s = torch.cross(t, n, dim=-1)

    vec = xyz[..., :1] * t + xyz[..., 1:2] * s + xyz[..., 2:] * n

    return vec


def pdf_uniform(w_i, normal, viewdirs, roughness_brdf, lgtSGs):
    return torch.tensor([1 / (2 * torch.pi)]).expand(normal.shape[0], normal.shape[1], 1)


def uniform_random_unit_hemisphere(base_shape, device):
    r1 = torch.rand(base_shape + (1,), device=device)  # [..., 1]
    r2 = torch.rand(base_shape + (1,), device=device)  # [..., 1]

    z = r1
    phi = 2 * torch.pi * r2
    x = phi.cos() * (1 - r1 ** 2).sqrt()
    y = phi.sin() * (1 - r1 ** 2).sqrt()

    return torch.cat([x, y, z], dim=-1)


def uniform_random_hemisphere(normal: torch.Tensor):
    """
    random uniform sample hemisphere of normal
    :param normal: [..., 3]; normal of surface
    :return [..., 3]
    """

    ray = uniform_random_unit_hemisphere(normal.shape[:-1], normal.device)
    ray = rotate_to_normal(ray, normal)

    return ray


def brdf_sampling(normal: torch.Tensor, roughness: torch.Tensor, viewdir: torch.Tensor):
    """
    :param normal: [..., 3]; normal of surface
    :param roughness: [..., 1]; roughness\
    :param viewdir: [..., 3]; w_o
    :return wi: [..., 3]
            pdf: [..., 1]
    """
    base_shape = normal.shape[:-1]
    device = normal.device

    # sampling h in unit coordinates
    r1 = torch.rand(base_shape + (1,), device=device)  # [..., 1]
    r2 = torch.rand(base_shape + (1,), device=device)  # [..., 1]

    theta = torch.arctan(roughness ** 2 * torch.sqrt(r1 / (1 - r1)))
    phi = 2 * torch.pi * r2

    z = theta.cos()
    y = theta.sin() * phi.sin()
    x = theta.sin() * phi.cos()

    h = torch.cat([x, y, z], dim=-1)

    # rotate to normal
    h = rotate_to_normal(h, normal)  # WARNNING there would be tinny change (about 1e-6) after rotating

    # convert to wi
    wi = 2 * (torch.sum(viewdir * h, dim=-1, keepdim=True)) * h - viewdir

    # # calculate pdf
    # # pdf_h = z * roughness ** 4 / torch.pi / (((roughness ** 4 - 1) * (z ** 2) + 1) ** 2)  # percision loss caused by float32 and result in nan
    # # pdf_h = z * roughness ** 4 / torch.pi / ((roughness ** 4 * (z ** 2) + (1 - z ** 2)) ** 2)
    # root = z ** 2 + (1 - z ** 2) / (roughness ** 4)
    # pdf_h = z / (torch.pi * (roughness ** 4) * root * root)
    #
    # h_dot_viewdir = torch.sum(h * viewdir, dim=-1, keepdim=True)
    # h_dot_viewdir = torch.clamp(h_dot_viewdir, min=TINY_NUMBER)
    # pdf_wi = pdf_h / (4 * h_dot_viewdir)

    pdf_wi = pdf_fn_brdf_gxx(wi, normal, viewdir, roughness, None)

    return wi, pdf_wi


def pdf_fn_brdf_gxx(wi, normal, viewdir, roughness, lgtSGs):
    h = wi + viewdir
    h = h / torch.norm(h, dim=-1, keepdim=True)
    # if wi = - viewdir, then their half vector should be normal or beyond semi-sphere, which would be rendered as zero afterwards
    mask = torch.isnan(h)
    h[mask] = normal[mask]

    cos_theta = torch.sum(h * normal, dim=-1, keepdim=True)
    cos_theta = torch.clamp(cos_theta, min=TINY_NUMBER)

    # pdf_h = cos_theta * roughness ** 4 / torch.pi / (((roughness ** 4 - 1) * (cos_theta ** 2) + 1) ** 2)  # percision loss caused by float32 and result in nan
    # pdf_h = cos_theta * roughness ** 4 / torch.pi / ((roughness ** 4 * (cos_theta ** 2) + (1 - cos_theta ** 2)) ** 2)
    root = cos_theta ** 2 + (1 - cos_theta ** 2) / (roughness ** 4)
    pdf_h = cos_theta / (torch.pi * (roughness ** 4) * root * root)

    h_dot_viewdir = torch.sum(h * viewdir, dim=-1, keepdim=True)
    h_dot_viewdir = torch.clamp(h_dot_viewdir, min=TINY_NUMBER)
    pdf_wi = pdf_h / (4 * h_dot_viewdir)

    return pdf_wi


def cos_sampling(normal: torch.Tensor):
    """
    :param normal: [..., 3]; normal of surface
    :return wi: [..., 3]
            pdf: [..., 1]
    """
    base_shape = normal.shape[:-1]
    device = normal.device

    # sampling h in unit coordinates
    r1 = torch.rand(base_shape + (1,), device=device)  # [..., 1]
    r2 = torch.rand(base_shape + (1,), device=device)  # [..., 1]

    theta = torch.arccos(torch.sqrt(1 - r1))
    phi = 2 * torch.pi * r2

    z = theta.cos()
    y = theta.sin() * phi.sin()
    x = theta.sin() * phi.cos()

    wi = torch.cat([x, y, z], dim=-1)

    # rotate to normal
    wi = rotate_to_normal(wi, normal)

    # calculate pdf
    pdf = z / torch.pi

    return wi, pdf


def pdf_fn_cos(wi, normal, viewdir, roughness, lgtSGs):
    cos_theta = torch.sum(wi * normal, dim=-1, keepdim=True)
    cos_theta = torch.clamp(cos_theta, min=TINY_NUMBER)

    pdf = cos_theta / torch.pi

    return pdf


def mix_sg_sampling(normal: torch.Tensor, lgtSGs: torch.Tensor):
    """
    mix gaussian sampling

    pdf(w_i) = sum_{k=1}^M alpha_k c_k exp{lambda_k(w_i cdot xi_k - 1)}
    alpha_k = frac{mu_k}{sum_{j=1}^M mu_j}
    1. sample based on alpha to decide use which single gaussian component
    2. sample w_i using the single gaussian component

    :param normal: [..., 3]; normal of surface
    :param lgtSGs: [M, 7]; sphere gaussian coefficient, [xi, lambda, mu]
    """
    base_shape = normal.shape[:-1]
    device = normal.device
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.unsqueeze(0).expand(base_shape + (M, 7))  # [dots_shape, M, 7]

    # unpack lgt sg coefficient
    xis = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    lambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
    mus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

    # compute \alpha_k = \frac{\mu_k}{\sum_{j=1}^M \mu_j}
    mus_energy = mus.sum(dim=-1, keepdim=True)  # [..., M, 1]
    # weight = mus_energy
    n_xi_dots = torch.sum(normal.unsqueeze(-2).expand(base_shape + (M, 3)) * xis, dim=-1, keepdim=True)  # [..., M, 1]
    weight = mus_energy * torch.clamp(n_xi_dots, TINY_NUMBER)  # give zero weight for the xi_k out of the hemisphere
    alpha = weight / weight.sum(dim=-2, keepdim=True)  # [..., M, 1]

    # sample gaussian component
    alpha_cumsum_right = torch.cumsum(alpha, dim=-2)  # [..., M, 1]
    alpha_cumsum_left = alpha_cumsum_right - alpha  # [..., M, 1]
    alpha_cumsum_right[..., -1, :] = 1.0  # numerical stable
    alpha_cumsum_left[..., 0, :] = 0.0  # numerical stable
    r0 = torch.rand(base_shape + (1, 1), device=device)  # [..., 1, 1]
    condition = (r0 >= alpha_cumsum_left) & (r0 < alpha_cumsum_right)  # [..., M, 1]

    try:
        xis_k = xis[condition.expand(xis.shape)].reshape(base_shape + (3,))  # [..., 3]
        lambdas_k = lambdas[condition].reshape(base_shape + (1,))  # [..., 1]
        mus_k = mus_energy[condition].reshape(base_shape + (1,))  # [..., 1]
    except:
        condition_num = condition.float()
        true_index = torch.max(condition_num, dim=-2, keepdim=True)[1]  # [..., 1, 1]

        xis_k = torch.gather(xis, dim=-2, index=true_index.expand(base_shape + (1, 3))).squeeze(-2)  # [..., 3]
        lambdas_k = torch.gather(lambdas, dim=-2, index=true_index).squeeze(-2)  # [..., 1]
        mus_k = torch.gather(mus, dim=-2, index=true_index).squeeze(-2)  # [..., 1]

    c_k = lambdas_k / (2 * torch.pi * (1 - torch.exp(-2 * lambdas_k)))  # [..., 1]

    # sample w_i based on k-th gaussian component
    r1 = torch.rand(base_shape + (1,), device=device)  # [..., 1]
    r2 = torch.rand(base_shape + (1,), device=device)  # [..., 1]

    theta = torch.arccos(
        1.0 / lambdas_k * torch.log(torch.clamp(
            1 - lambdas_k * r1 / (2 * torch.pi * c_k)
            , TINY_NUMBER))  # for numeric stable
        + 1
    )
    phi = 2 * torch.pi * r2

    z = theta.cos()
    y = theta.sin() * phi.sin()
    x = theta.sin() * phi.cos()

    wi = torch.cat([x, y, z], dim=-1)  # [..., 3]

    # rotate to xis_k
    wi = rotate_to_normal(wi, xis_k)

    # calculate pdf
    pdf_wi = pdf_fn_mix_sg(wi, normal, None, None, lgtSGs)

    return wi, pdf_wi


def pdf_fn_mix_sg(wi, normal, viewdir, roughness, lgtSGs):
    base_shape = normal.shape[:-1]
    M = lgtSGs.shape[-2]
    device = normal.device

    # unpack lgt sg coefficient
    xis = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    lambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
    mus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

    # compute \alpha_k = \frac{\mu_k}{\sum_{j=1}^M \mu_j}
    mus_energy = mus.sum(dim=-1, keepdim=True)  # [..., M, 1]
    # weight = mus_energy
    n_xi_dots = torch.sum(normal.unsqueeze(-2).expand(base_shape + (M, 3)) * xis, dim=-1, keepdim=True)  # [..., M, 1]
    weight = mus_energy * torch.clamp(n_xi_dots, TINY_NUMBER)  # give zero weight for the xi_k out of the hemisphere
    alpha = weight / weight.sum(dim=-2, keepdim=True)  # [..., M, 1]

    # compute c_k
    c = lambdas / (2 * torch.pi * (1 - torch.exp(-2.0 * lambdas)))  # [..., M, 1]

    # compute pdf
    wi = wi.unsqueeze(-2).expand(base_shape + (M, 3))  # [..., M, 3]
    dots = torch.sum(wi * xis, dim=-1, keepdim=True)  # [..., M, 1]
    pdf_wi = alpha * c * torch.exp(lambdas * (dots - 1))  # [..., M, 1]
    pdf_wi = pdf_wi.sum(dim=-2)  # [..., 1]

    return pdf_wi


def render_with_BRDF_multiple_sampling(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs,
                                       points, sdf_network, color_network, ray_tracer, nlights=64,
                                       blending_weights=None, diffuse_rgb=None, diff_geo=True, speed_first=False):
    """
    render in path tracing style.
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [..., 3]; / [1, 3] when fix_specular
    :param roughness: [..., 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param points: [..., 3]; render position on surface
    :param model: implicit_differentiable_renderer
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    """

    nrays = normal.shape[0]

    viewdirs = viewdirs.unsqueeze(1).expand(nrays, nlights, 3)  # [nrays,nlights,3]
    normal = normal.unsqueeze(1).expand(nrays, nlights, 3)
    roughness = roughness.unsqueeze(1).expand(nrays, nlights, 1)
    diffuse_albedo = diffuse_albedo.unsqueeze(1).expand(nrays, nlights, 3).reshape(-1, 3)
    points = points.unsqueeze(1).expand(nrays, nlights, 3)

    wi_sampled = []
    specular_rgb_final = 0
    diffuse_rgb_final = 0
    rgb_final = 0
    indir_rgb_final = 0
    vis_final = 0

    with torch.no_grad():
        # # uniform sample
        # w_i = uniform_random_hemisphere(normal)  # random reflect from a direction in hemisphere
        # pdf = torch.tensor([1 / (2 * torch.pi)]).expand(normal.shape[0],normal.shape[1],1)
        # wi_sampled.append((w_i, pdf,pdf_uniform))

        # cos weighted sampling
        w_i, pdf = cos_sampling(normal)
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_cos))

        # brdf weighted sampling
        roughness_brdf = roughness  # [..., 1]
        w_i, pdf = brdf_sampling(normal, roughness_brdf, viewdirs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_brdf_gxx))

        # mix sg weighted sampling
        w_i, pdf = mix_sg_sampling(normal, lgtSGs)  # [..., 3], [..., 1]
        pdf = torch.clamp(pdf, min=TINY_NUMBER)
        wi_sampled.append((w_i, pdf, pdf_fn_mix_sg))

        ### prepare for multi importance sampling
        n_list = [1] * len(wi_sampled)
        pdf_list = list(zip(*wi_sampled))[1]
        pdf_array = []
        for i in range(len(wi_sampled)):
            pdf_array_i = []
            w_i = wi_sampled[i][0]
            for j in range(len(wi_sampled)):
                if j == i:
                    pdf_array_i.append(pdf_list[i])
                else:
                    pdf_fn = wi_sampled[j][2]
                    pdf_array_i.append(pdf_fn(w_i, normal, viewdirs, roughness_brdf, lgtSGs))
            pdf_array.append(pdf_array_i)

    visible_list = []
    indirect_light_list = []

    for sample_type_index, (w_i, _, _) in enumerate(wi_sampled):
        visibility, indirect_light = compute_secondary_shading_effects(
            sdf_network=sdf_network,
            color_network=color_network,
            ray_tracer=ray_tracer,
            surface_pts=points.reshape(-1, 3),
            surf2light=w_i.reshape(-1, 3),
            chunk_size=500000
        )

        visible_list.append(visibility.reshape(-1, 1))
        indirect_light_list.append(indirect_light.reshape(-1, 3))

    M = lgtSGs.shape[0]
    dots_shape = list(normal.reshape(-1, 3).shape[:-1])

    # shape align
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]

    for sample_type_index, (w_i, pdf, _) in enumerate(wi_sampled):
        visibility = visible_list[sample_type_index]  # [..., 1]
        indirect_light = indirect_light_list[sample_type_index]  # [..., 3]

        # source light
        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])  # [..., M, 1]
        lgtSGMus = torch.abs(lgtSGs[..., -3:])  # [..., M, 3] positive values

        w_i_light = w_i.reshape(-1, 3).unsqueeze(-2).expand(dots_shape + [M, 3])  # [..., M, 3]
        light = sg_fn(w_i_light, lgtSGLobes, lgtSGLambdas, lgtSGMus)  # [..., M, 3]
        light = light.sum(-2)  # [..., 3]

        light_rgbs = visibility * light + indirect_light * (
                1 - visibility)  # [bs, envW * envH, 3]

        # fs
        viewdirs_fs = viewdirs.reshape(-1, 3)
        normal_fs = normal.reshape(-1, 3)
        w_i_fs = w_i.reshape(-1, 3)

        half_fs = w_i_fs + viewdirs_fs  # [..., 3]
        half_fs = half_fs / (torch.norm(half_fs, dim=-1, keepdim=True) + TINY_NUMBER)

        # NDF - GGX
        n_dot_h = torch.sum(normal_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        n_dot_h = torch.clamp(n_dot_h, min=0)
        roughness = roughness.reshape(-1, 1)
        roughness_pow2 = roughness ** 2  # [..., 1]
        # D = roughness_pow2 ** 2 / torch.pi / ((n_dot_h ** 2 * (roughness_pow2 ** 2 - 1) + 1) ** 2)
        root = n_dot_h ** 2 + (1 - n_dot_h ** 2) / (roughness_pow2 ** 2)
        D = 1.0 / (torch.pi * (roughness_pow2 ** 2) * root * root)

        # fresnel terms
        v_dot_h = torch.sum(viewdirs_fs * half_fs, dim=-1, keepdim=True)  # [..., 1]
        v_dot_h = torch.clamp(v_dot_h, min=0.)  # note: for numeric stability
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)
        #
        # geometric terms
        dot1 = torch.sum(viewdirs_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_o, n>
        dot1 = torch.clamp(dot1, min=0.)  # note: for numeric stability
        dot2 = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # equals <w_i, n>
        dot2 = torch.clamp(dot2, min=0.)  # note: for numeric stability
        k = ((roughness + 1.) * (roughness + 1.) / 8.)
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        fs = F * D * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [..., 3]

        # weight
        weight = power_heuristic_list(n_list, pdf_array[sample_type_index], sample_type_index).reshape(-1, 1)

        # specular rgb
        w_i_dot_normal = torch.sum(w_i_fs * normal_fs, dim=-1, keepdim=True)  # [..., 1]
        w_i_dot_normal = torch.clamp(w_i_dot_normal, min=0)  # NOTE wi should not beyond hemisphere

        specular_rgb = weight * light_rgbs * fs * w_i_dot_normal / pdf.reshape(-1, 1)  # [..., 3]
        specular_rgb = torch.clamp(specular_rgb, min=0.)
        specular_rgb = specular_rgb.reshape(nrays, nlights, 3)
        specular_rgb = torch.mean(specular_rgb, dim=1)

        # diffuse rgb
        # multiply with light
        diffuse_rgb = weight * light_rgbs * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf.reshape(-1, 1)  # [..., 3]
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)
        diffuse_rgb = diffuse_rgb.reshape(nrays, nlights, 3)
        diffuse_rgb = torch.mean(diffuse_rgb, dim=1)

        indir_light = (1 - visibility) * indirect_light

        indir_specular_rgb = weight * indir_light * fs * w_i_dot_normal / pdf.reshape(-1, 1)  # [..., 3]
        indir_specular_rgb = torch.clamp(indir_specular_rgb, min=0.)
        indir_specular_rgb = indir_specular_rgb.reshape(nrays, nlights, 3)
        indir_specular_rgb = torch.mean(indir_specular_rgb, dim=1)

        # diffuse rgb
        # multiply with light
        indir_diffuse_rgb = weight * indir_light * (diffuse_albedo / np.pi) * w_i_dot_normal / pdf.reshape(-1,
                                                                                                           1)  # [..., 3]
        indir_diffuse_rgb = torch.clamp(indir_diffuse_rgb, min=0.)
        indir_diffuse_rgb = indir_diffuse_rgb.reshape(nrays, nlights, 3)
        indir_diffuse_rgb = torch.mean(indir_diffuse_rgb, dim=1)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + diffuse_rgb
        indir_rgb = indir_specular_rgb + indir_diffuse_rgb

        specular_rgb_final += specular_rgb
        diffuse_rgb_final += diffuse_rgb
        rgb_final += rgb
        indir_rgb_final += indir_rgb
        vis = torch.mean(visibility.reshape(nrays, nlights, 1), dim=1)
        vis_final += vis

    ret = {'sg_rgb': rgb_final,
           'indir_rgb': indir_rgb_final,
           'vis': vis_final / 3,
           'specular_rgb': specular_rgb_final,
           'diffuse_rgb': diffuse_rgb_final,
           'sg_diffuse_albedo': diffuse_albedo
           }

    return ret


def power_heuristic_list(n_list, pdf_list, index):
    cur = (n_list[index] * pdf_list[index]) ** 2
    all_sum = 0
    for i in range(len(n_list)):
        all_sum += (n_list[i] * pdf_list[i]) ** 2

    if type(all_sum) == torch.Tensor:
        all_sum = torch.clamp(all_sum, min=TINY_NUMBER)
    else:
        all_sum = max(all_sum, TINY_NUMBER)

    return cur / all_sum
