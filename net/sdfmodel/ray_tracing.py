import torch
import torch.nn as nn


def get_sphere_intersection(rays_o, rays_d, r=1.0):
    """
    input:
        rays_o:batch_size,3
        rays_d:batch_size,3
    return:
        sphere_intersections：batch_size,2(t_near,t_far/0,0)
        mask_intersect: batch_size(0/1)
    """
    batch_size = rays_o.shape[0]
    a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    c = torch.sum(rays_o ** 2, dim=-1, keepdim=True) - r ** 2
    under_sqrt = b ** 2 - 4 * a * c
    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros(batch_size, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor(
        [-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= b[mask_intersect]
    sphere_intersections = sphere_intersections / 2.0
    sphere_intersections = sphere_intersections.clamp_min(0.01)

    return sphere_intersections, mask_intersect


class RayTracing(nn.Module):
    def __init__(
            self,
            object_bounding_sphere=1.0,
            sdf_threshold=5.0e-5,
            line_search_step=0.5,
            line_step_iters=1,
            sphere_tracing_iters=10,
            n_steps=100,
            n_rootfind_steps=8,
    ):
        super().__init__()

        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_rootfind_steps

    def forward(self,
                sdf,
                cam_loc,
                object_mask,
                ray_directions
                ):

        batch_size = ray_directions.shape[0]

        sphere_intersections, mask_intersect = get_sphere_intersection(cam_loc, ray_directions,
                                                                       r=self.object_bounding_sphere)
        far = sphere_intersections[:, 1]
        near = sphere_intersections[:, 0]
        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(batch_size, sdf, cam_loc, ray_directions, near, far, mask_intersect)
        network_object_mask = (acc_start_dis < acc_end_dis)

        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start
        # sampler_mask = unfinished_mask_start | (~network_object_mask)
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        if sampler_mask.sum() > 0:
            sampler_min_max = torch.zeros((batch_size, 2)).cuda()
            sampler_min_max[sampler_mask, 0] = acc_start_dis[sampler_mask]
            sampler_min_max[sampler_mask, 1] = acc_end_dis[sampler_mask]

            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                cam_loc,
                                                                                object_mask,
                                                                                ray_directions,
                                                                                sampler_min_max,
                                                                                sampler_mask
                                                                                )

            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

        # print('----------------------------------------------------------------')
        # print('RayTracing: object = {0}/{1}, rootfind on {2}/{3}.'
        #       .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        # print('----------------------------------------------------------------')

        # if not self.training:
        #     return curr_start_points, \
        #            network_object_mask, \
        #            acc_start_dis
        #
        # ray_directions = ray_directions.reshape(-1, 3)
        # mask_intersect = mask_intersect.reshape(-1)
        #
        # in_mask = ~network_object_mask & object_mask & ~sampler_mask
        # out_mask = ~object_mask & ~sampler_mask
        #
        # mask_left_out = (in_mask | out_mask) & ~mask_intersect
        # if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
        #     cam_left_out = cam_loc[mask_left_out]
        #     rays_left_out = ray_directions[mask_left_out]
        #     acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
        #     curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out
        #
        # mask = (in_mask | out_mask) & mask_intersect
        #
        # if mask.sum() > 0:
        #     min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]
        #
        #     min_mask_points, min_mask_dist = self.minimal_sdf_points(batch_size, sdf, cam_loc, ray_directions, mask, min_dis, max_dis)
        #
        #     curr_start_points[mask] = min_mask_points
        #     acc_start_dis[mask] = min_mask_dist

        return curr_start_points, \
            network_object_mask, \
            acc_start_dis

    def sphere_tracing(self, n_rays, sdf, rays_o, rays_d, near, far, mask_intersect):
        """
            input:
            n_rays:n_rays
            rays_o:n_rays,3
            rays_d:n_rays,3
            mask_intersect:n_rays,(True)
        """
        near_points = rays_o + near.reshape(-1, 1) * rays_d  # n_rays,3
        far_points = rays_o + far.reshape(-1, 1) * rays_d  # n_rays,3
        # sphere_intersections_points = rays_o[:, None, :] + sphere_intersections[..., :, None]  * rays_d[:, None, :]    #n_rays,2,3

        unfinished_mask_start = mask_intersect.reshape(-1).clone()  # 一串 n_rays,
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        # Initialize start current points
        curr_start_points = torch.zeros(n_rays, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = near_points.reshape(-1, 3)[
            unfinished_mask_start]  # near_point

        acc_start_dis = near.reshape(-1)
        acc_start_dis[unfinished_mask_start] = (near.reshape(-1))[unfinished_mask_start]  # t_near

        # Initialize end current points
        curr_end_points = torch.zeros(n_rays, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = far_points.reshape(-1, 3)[
            unfinished_mask_end]  # far_point
        acc_end_dis = far.reshape(-1)
        acc_end_dis[unfinished_mask_end] = (far.reshape(-1))[unfinished_mask_end]  # t_far

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()  # t_near
        max_dis = acc_end_dis.clone()  # t_far

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start]).squeeze()  # n_rays,

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end]).squeeze()

        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)

            if (
                    unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break

            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start  # t_near
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = (rays_o + acc_start_dis.reshape(-1, 1) * rays_d).reshape(
                -1, 3)

            curr_end_points = (rays_o + acc_end_dis.reshape(-1, 1) * rays_d).reshape(
                -1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start]).squeeze()

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end]).squeeze()

            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (
                    not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * \
                                                      curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (rays_o + acc_start_dis.reshape(-1, 1) * rays_d).reshape(
                    -1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[
                    not_projected_end]
                curr_end_points[not_projected_end] = (rays_o + acc_end_dis.reshape(-1, 1) * rays_d).reshape(
                    -1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start]).squeeze()
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end]).squeeze()

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def ray_sampler(self, sdf, rays_o, object_mask, rays_d, sampler_min_max, sampler_mask):
        ''' Sample the ray in a given range and run rootfind on rays which have sign transition '''

        batch_size = rays_o.shape[0]
        n_total_pxl = batch_size
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_pts_end = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()
        sampler_dists_end = torch.zeros(n_total_pxl).cuda().float()

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda()

        pts_intervals = sampler_min_max[:, 0].unsqueeze(-1) + intervals_dist * (
                sampler_min_max[:, 1] - sampler_min_max[:, 0]).unsqueeze(-1)
        points = rays_o[:, None, :] + rays_d[:, None, :] * pts_intervals[..., :, None]

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()  # 非零元素的索引
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape(
            (1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)  # 返回最小值的索引
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        sampler_pts_end_ind = torch.argmax(tmp, -1)  # 返回最大值的索引
        sampler_pts_end[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_end_ind, :]
        sampler_dists_end[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_end_ind]

        true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels（没有交点的射线取最小sdf值的点（离表面最近的点）
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx,
                                                          :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][
                torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]  # n_secant_pts，
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            rays_o_secant = rays_o[
                mask_intersect_idx[secant_pts]]
            rays_d_secant = rays_d.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, rays_o_secant, rays_d_secant, sdf)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = rays_o_secant + z_pred_secant.reshape(-1, 1) * rays_d_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''
        eps = 1.e-8
        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low + eps) + z_low
        z_pred = z_pred.clamp(0., 2e1)

        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.reshape(-1, 1) * ray_directions
            sdf_mid = sdf(p_mid).squeeze()
            ind_low = (sdf_mid > 0).squeeze()
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = (sdf_mid < 0).squeeze()
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low + eps) + z_low
            z_pred = z_pred.clamp(0., 2e1)

        return z_pred

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''

        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = torch.linspace(0.0, 1.0,n).cuda()
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = cam_loc[mask]
        mask_rays = ray_directions[mask]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist
