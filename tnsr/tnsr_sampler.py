"""
Transparent Neural Surface Refinement implementation
"""

from typing import Callable, Optional, Tuple, List

import torch
from torch import nn
import copy

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.utils.math import intersect_aabb
from nerfstudio.model_components.ray_samplers import Sampler, PDFSampler, UniformLinDispPiecewiseSampler

"""
Transparent Neural Surface Refinement implementation
"""


def points_in_3d_cuboid(points, min_corner, max_corner):
    """
    Check whether multiple 3D points are inside a cuboid.

    Args:
        points: Coordinates of multiple 3D points, a torch.Tensor of shape (N, 3).
        min_corner: Minimum corner of the cuboid, a torch.Tensor of shape (3,).
        max_corner: Maximum corner of the cuboid, a torch.Tensor of shape (3,).

    Returns:
        A list of boolean values, indicating whether each point is inside the cuboid.
    """
    return ((min_corner <= points) & (points <= max_corner)).all(-1)


class ProposalNetworkSampler(Sampler):
    """Sampler that uses a proposal network to generate samples.

    Args:
        num_proposal_samples_per_ray: Number of samples to generate per ray for each proposal step.
        num_nerf_samples_per_ray: Number of samples to generate per ray for the NERF model.
        num_proposal_network_iterations: Number of proposal network iterations to run.
        single_jitter: Use a same random jitter for all samples along a ray.
        update_sched: A function that takes the iteration number of steps between updates.
        initial_sampler: Sampler to use for the first iteration. Uses UniformLinDispPiecewise if not set.
        pdf_sampler: PDFSampler to use after the first iteration. Uses PDFSampler if not set.
    """

    def __init__(
            self,
            num_proposal_samples_per_ray: Tuple[int, ...] = (64,),
            num_nerf_samples_per_ray: int = 32,
            num_proposal_network_iterations: int = 2,
            single_jitter: bool = False,
            update_sched: Callable = lambda x: 1,
            initial_sampler: Optional[Sampler] = None,
            pdf_sampler: Optional[PDFSampler] = None,
    ) -> None:
        super().__init__()
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")

        # samplers
        if initial_sampler is None:
            self.initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            self.initial_sampler = initial_sampler
        if pdf_sampler is None:
            self.pdf_sampler = PDFSampler(include_original=False, single_jitter=single_jitter)
        else:
            self.pdf_sampler = pdf_sampler

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def set_anneal(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._anneal = anneal

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self._steps_since_update += 1

    def generate_ray_samples(
            self,
            ray_bundle: Optional[RayBundle] = None,
            density_fns: Optional[List[Callable]] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](ray_samples.frustums.get_positions())
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.frustums.get_positions())
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list

    def find_surface_points(self, ray_samples, sdf_fn):
        """
        Compute the surface points using linear interpolation based on sign changes in the SDF.

        Parameters:
        - ray_samples: A data structure with information about the ray samples.
            It should have fields 'frustums.starts', 'frustums.origins', and 'frustums.directions'.
        - sdf_fn: A function that returns the signed distance field values for the ray samples.

        Returns:
        - surface_points: A tensor representing the computed surface points.
        """

        with torch.no_grad():
            sdf = sdf_fn(ray_samples)

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        n_rays, n_samples = ray_samples.shape
        starts = ray_samples.frustums.starts
        sign_matrix = torch.cat(
            [torch.sign(sdf[:, :-1, 0] * sdf[:, 1:, 0]), torch.ones(n_rays, 1).to(sdf.device)], dim=-1
        )
        cost_matrix = sign_matrix * torch.arange(n_samples, 0, -1).float().to(sdf.device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_pos_to_neg = sdf[torch.arange(n_rays), indices, 0] > 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_pos_to_neg

        # Get depth values and function values for the interval
        d_low = starts[torch.arange(n_rays), indices, 0][mask]
        v_low = sdf[torch.arange(n_rays), indices, 0][mask]

        indices = torch.clamp(indices + 1, max=n_samples - 1)
        d_high = starts[torch.arange(n_rays), indices, 0][mask]
        v_high = sdf[torch.arange(n_rays), indices, 0][mask]

        # linear-interpolations
        z = (v_low * d_high - v_high * d_low) / (v_low - v_high)

        origins = ray_samples.frustums.origins[torch.arange(n_rays), indices, :][mask]
        directions = ray_samples.frustums.directions[torch.arange(n_rays), indices, :][mask]
        surface_points = origins + directions * z[..., None]

        if surface_points.shape[0] <= 0:
            surface_points = torch.rand((1024, 3), device=sdf.device) - 0.5

        return surface_points


class TraceModule(nn.Module):
    """
    Trace Module class.

    The trace module is a wrapper class for the autograd function
    DepthFunction (see below).

    """

    def __init__(self, sphere_radius=0.5, sphere_tracing_iters=15, sdf_threshold=5.0e-5, n_steps=100,
                 line_step_iters=1, line_search_step=0.5, n_secant_steps=8, eta=1.45):
        super().__init__()
        self.sphere_radius = sphere_radius
        self.sphere_tracing_iters = sphere_tracing_iters
        self.sdf_threshold = sdf_threshold
        self.n_steps = n_steps
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_secant_steps = n_secant_steps
        self.cal_grad = GradFunction.apply
        self.eta = eta

    def forward(self, ray_bundle: Optional[RayBundle] = None,
                sdf_pts_fn: Optional[Callable] = None,
                cuboid_coordinates: Tuple = (-0.15, -0.15, -0.15, 0.15, 0.15, 0.15)
                ):

        # first intersection
        ray_bundle = copy.deepcopy(ray_bundle)
        ray_bundle_refl = copy.deepcopy(ray_bundle)

        sphere_intersections, mask_intersect = self.get_cuboid_intersection(ray_bundle, cuboid_coordinates)

        if mask_intersect.sum() > 0:
            valid_mask, intersection_depth, normals = self.intersection_with_sdf(ray_bundle, mask_intersect,
                                                                                 sphere_intersections, sdf_pts_fn,
                                                                                 far=False)
            if valid_mask.sum() > 0:
                normals = normals.to(ray_bundle.origins)
                normals = normals[valid_mask]

                valid_mask = valid_mask.to(ray_bundle.origins.device)
                hit_rays = copy.deepcopy(ray_bundle[valid_mask])

                init_depth = intersection_depth[valid_mask].to(ray_bundle.origins.device)
                in_dir = hit_rays.directions
                first_surface_points = hit_rays.origins + init_depth.unsqueeze(1) * hit_rays.directions

                # ------------------ gradient calculating ------------------ #
                s1 = sdf_pts_fn(first_surface_points)
                input = [s1, first_surface_points, in_dir, normals, init_depth]
                init_depth = self.cal_grad(*input)
                first_surface_points = hit_rays.origins + init_depth.unsqueeze(1) * hit_rays.directions
                # ------------------ gradient calculating ------------------ #

                inside_mask = points_in_3d_cuboid(points=first_surface_points,
                                                  min_corner=torch.Tensor([cuboid_coordinates[:3]]).to(
                                                      first_surface_points),
                                                  max_corner=torch.Tensor([cuboid_coordinates[3:]]).to(
                                                      first_surface_points)
                                                  )

                # first refraction
                l_t1, attenuate1, totalReflectMask1 = self.refraction(in_dir, normals, eta1=1.0003, eta2=self.eta)
                l_r1 = self.reflection(in_dir, normals)

                hit_rays.origins = first_surface_points - 1 * l_t1
                hit_rays.directions = l_t1

                # second intersection
                sphere_intersections, mask_intersect_ = self.get_cuboid_intersection(hit_rays, cuboid_coordinates)
                mask_intersect_ = mask_intersect_ & inside_mask

                if mask_intersect_.sum() > 0:
                    valid_mask_, intersection_depth_, normals_ = self.intersection_with_sdf(hit_rays, mask_intersect_,
                                                                                            sphere_intersections,
                                                                                            sdf_pts_fn, far=True)

                    # TODO
                    # dealing with rays missing the second intersection (unwatertight surface)
                    # invalid_mask_ = (valid_mask_ == False)
                    # refr_rays_ = self.copy_index(refr_rays_, invalid_mask_, hit_rays[invalid_mask_])

                    if valid_mask_.sum() > 0:
                        normals_ = normals_.to(hit_rays.origins.device)
                        normals_ = normals_[valid_mask_]

                        valid_mask_ = valid_mask_.to(hit_rays.origins.device)

                        init_depth_ = intersection_depth_[valid_mask_].to(hit_rays.origins.device)

                        second_surface_points = hit_rays.origins[valid_mask_] + init_depth_.unsqueeze(1) * \
                                                hit_rays.directions[valid_mask_]
                        in_dir_ = hit_rays.directions[valid_mask_]

                        selected_indices_I = torch.where(valid_mask)[0]
                        selected_indices_final = selected_indices_I[valid_mask_]

                        # ------------------ gradient calculating ------------------ #
                        s2 = sdf_pts_fn(second_surface_points)
                        input = [s2, second_surface_points, in_dir_, normals_, init_depth_]
                        init_depth_ = self.cal_grad(*input)
                        second_surface_points = hit_rays.origins[valid_mask_] + init_depth_.unsqueeze(1) * \
                                                hit_rays.directions[
                                                    valid_mask_]
                        # ------------------ gradient calculating ------------------ #

                        # second refraction
                        l_t2, attenuate2, totalReflectMask2 = self.refraction(in_dir_, -normals_, eta1=self.eta,
                                                                              eta2=1.0003)

                        ray_bundle.origins[selected_indices_final] = second_surface_points + 0.1 * l_t2
                        ray_bundle.directions[selected_indices_final] = l_t2

                        ray_bundle_refl.origins[selected_indices_final] = first_surface_points[
                                                                              valid_mask_].detach() + 0.1 * \
                                                                          l_r1[valid_mask_]

                        ray_bundle_refl.directions[selected_indices_final] = l_r1[valid_mask_]

                        return ray_bundle, selected_indices_final, ray_bundle_refl, attenuate1[
                            valid_mask_], first_surface_points.detach(), second_surface_points.detach()

            # If none of the conditions above are met, return an empty list.
            return ray_bundle, [], [], [], [], []

        # If none of the conditions above are met, return an empty list.
        return ray_bundle, [], [], [], [], []

    def intersection_with_sdf(self, ray_bundle, mask_intersect, sphere_intersections, sdf_pts_fn, far=False):
        with torch.no_grad():
            B = ray_bundle.shape[0]
            (
                curr_start_points,
                curr_end_points,
                unfinished_mask_start,
                unfinished_mask_end,
                acc_start_dis,
                acc_end_dis,
                min_dis,
                max_dis,
            ) = self.sphere_tracing(ray_bundle, mask_intersect, sphere_intersections, sdf_pts_fn)

            # hit and non convergent rays
            network_object_mask = acc_start_dis < acc_end_dis
            # The non convergent rays should be handled by the sampler
            sampler_mask = unfinished_mask_start if not far else unfinished_mask_end
            sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
            if sampler_mask.sum() > 0:
                sampler_min_max = torch.zeros((B, 2)).cuda()
                sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[sampler_mask]
                sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]

                sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(ray_bundle, sampler_min_max,
                                                                                    sampler_mask, sdf_pts_fn, far=far)
                if not far:
                    curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
                    acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
                else:
                    curr_end_points[sampler_mask] = sampler_pts[sampler_mask]
                    acc_end_dis[sampler_mask] = sampler_dists[sampler_mask]
                network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

            valid_mask = network_object_mask
            intersection_depth = acc_start_dis if not far else acc_end_dis

        # access normals
        hit_points = curr_start_points if not far else curr_end_points
        with torch.enable_grad():
            hit_points = hit_points.clone().detach().requires_grad_(True)
            pred = sdf_pts_fn(hit_points)
            hit_points_sdf = pred
            d_points = torch.ones_like(
                hit_points_sdf, requires_grad=False, device=hit_points_sdf.device
            )
            hit_points_sdf.requires_grad_(True)
        verts_grad = torch.autograd.grad(
            outputs=hit_points_sdf,
            inputs=hit_points,
            grad_outputs=d_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        normals = torch.zeros(B, 3).cuda()
        normals[valid_mask] = verts_grad[valid_mask].to(device=normals.device)
        normals[valid_mask] = normals[valid_mask] / normals[valid_mask].norm(2, dim=1).unsqueeze(-1)
        normals_ = normals.detach()
        # normals_ = normals # will introduce some noisy gradients

        return valid_mask, intersection_depth, normals_

    def get_cuboid_intersection(self, rays, cuboid_coordinates):
        cam_loc = rays.origins
        ray_directions = rays.directions

        # Check if the point is inside or outside the cuboid
        bbox = torch.tensor(cuboid_coordinates).to(rays.origins)
        t_min, t_max = intersect_aabb(cam_loc, ray_directions, bbox)
        sphere_intersections = torch.cat([t_min.unsqueeze(-1), t_max.unsqueeze(-1)], dim=1)
        mask_intersect = t_min < 1e10
        return sphere_intersections, mask_intersect

    def sphere_tracing(
            self,
            rays,
            mask_intersect,
            sphere_intersections,
            sdf
    ):
        """Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection"""
        B = rays.shape[0]
        cam_loc = rays.origins
        ray_directions = rays.directions
        sphere_intersections_points = sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(
            1) + cam_loc.unsqueeze(1)
        unfinished_mask_start = mask_intersect.clone()
        unfinished_mask_end = mask_intersect.clone()

        # Initialize start current points
        curr_start_points = torch.zeros(B, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:, 0, :].reshape(-1, 3)[
            unfinished_mask_start]

        acc_start_dis = torch.zeros(B).cuda().float()
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1, 2)[unfinished_mask_start, 0]

        # Initialize end current points
        curr_end_points = torch.zeros(B, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:, 1, :].reshape(-1, 3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(B).cuda().float()
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1, 2)[unfinished_mask_end, 1]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

        # assert torch.all(next_sdf_start >= 0) and torch.all(next_sdf_end >= 0)
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
                    unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0
            ) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = cam_loc + acc_start_dis.unsqueeze(1) * ray_directions
            curr_end_points = cam_loc + acc_end_dis.unsqueeze(1) * ray_directions

            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            if unfinished_mask_start.sum() > 0:
                next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            if unfinished_mask_end.sum() > 0:
                next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

            # Fix points which wrongly crossed the surface
            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (
                    not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * \
                                                      curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc + acc_start_dis.unsqueeze(1) * ray_directions)[
                    not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[
                    not_projected_end]
                curr_end_points[not_projected_end] = (cam_loc + acc_end_dis.unsqueeze(1) * ray_directions)[
                    not_projected_end]

                # Calc sdf
                if not_projected_start.sum() > 0:
                    next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                if not_projected_end.sum() > 0:
                    next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end])

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return (
            curr_start_points, curr_end_points, unfinished_mask_start, unfinished_mask_end, acc_start_dis, acc_end_dis,
            min_dis, max_dis)

    def ray_sampler(self, rays, sampler_min_max, sampler_mask, sdf, far=False):
        """Sample the ray in a given range and run secant on rays which have sign transition"""
        B = rays.shape[0]
        cam_loc = rays.origins
        ray_directions = rays.directions

        sampler_pts = torch.zeros(B, 3).cuda().float()
        sampler_dists = torch.zeros(B).cuda().float()

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda()
        pts_intervals = sampler_min_max[:, 0].unsqueeze(-1) + intervals_dist * (
                sampler_min_max[:, 1] - sampler_min_max[:, 0]).unsqueeze(-1)
        points = cam_loc.reshape(B, 1, 3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(1)

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape(
            (1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1) if not far else torch.argmax(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        net_surface_pts = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0

        # take points with minimal SDF value for P_out pixels
        p_out_mask = ~net_surface_pts
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
        secant_pts = net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[
                torch.arange(pts_intervals.shape[0]), sampler_pts_ind
            ][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][
                secant_pts
            ]
            z_low = pts_intervals[secant_pts][
                torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1
            ]
            sdf_low = sdf_val[secant_pts][
                torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1
            ]
            cam_loc_secant = (
                cam_loc.unsqueeze(1)
                    .reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            )
            ray_directions_secant = ray_directions.reshape((-1, 3))[
                mask_intersect_idx[secant_pts]
            ]
            z_pred_secant = self.secant(
                sdf_low,
                sdf_high,
                z_low,
                z_high,
                cam_loc_secant,
                ray_directions_secant,
                sdf
            )
            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = (
                    cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant)
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant
        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        """Runs the secant method for interval [z_low, z_high] for n_secant_steps"""
        z_pred = -sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]
            z_pred = -sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        return z_pred

    def adjust_normal(self, normals, in_dir):
        in_dot = (in_dir * normals).sum(dim=-1)
        mask = in_dot > 0
        normals[mask] = -normals[mask]  # make sure normal point to in_dir
        return normals

    def refraction(self, l, normal, eta1, eta2):
        # l n x 3 x imHeight x imWidth
        # normal n x 3 x imHeight x imWidth
        # eta1 float
        # eta2 float
        cos_theta = torch.sum(l * (-normal), dim=1).unsqueeze(1)  # [10, 1, 192, 256]
        i_p = l + normal * cos_theta
        t_p = eta1 / eta2 * i_p

        t_p_norm = torch.sum(t_p * t_p, dim=1)
        totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(1)

        t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999)).unsqueeze(1).expand_as(normal) * (-normal)
        t = t_i + t_p
        t = t / torch.sqrt(torch.clamp(torch.sum(t * t, dim=1), min=1e-10)).unsqueeze(1)

        cos_theta_t = torch.sum(t * (-normal), dim=1).unsqueeze(1)

        e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
              torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
        e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
              torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)

        attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1).detach()

        return t, attenuate, totalReflectMask

    def reflection(self, l, normal):
        # l n x 3 x imHeight x imWidth
        # normal n x 3 x imHeight x imWidth
        # eta1 float
        # eta2 float
        cos_theta = torch.sum(l * (-normal), dim=1).unsqueeze(1)
        r_p = l + normal * cos_theta
        r_p_norm = torch.clamp(torch.sum(r_p * r_p, dim=1), 0, 0.999999)
        r_i = torch.sqrt(1 - r_p_norm).unsqueeze(1).expand_as(normal) * normal
        r = r_p + r_i
        r = r / torch.sqrt(torch.clamp(torch.sum(r * r, dim=1), min=1e-10).unsqueeze(1))

        return r

    def copy_index(self, inputs, mask, src):
        # Find the indices where mask is True
        index = torch.nonzero(mask).reshape(-1).long()
        # Create a copy of the inputs tensor and apply index_copy
        outputs = inputs.clone()
        outputs = outputs.view(-1, *inputs.shape[1:])
        outputs.index_copy_(0, index, src)

        return outputs.view_as(inputs)


class GradFunction(torch.autograd.Function):
    ''' Sphere Trace Function class.

    It provides the function to march along given rays to detect the surface
    points for the SDF Network.

    The backward pass is implemented using
    the analytic gradient described in the publication CVPR 2020.
    '''

    @staticmethod
    def forward(ctx, *input):
        ''' Performs a forward pass of the Depth function.

        s, delta, do, n

        return delta
        Args:
            input (list): input to forward function
        '''
        s, x, d, n, delta = input
        # sdf of first intersection, first_intersection, in-direction, normal, delta

        # Save values for backward pass
        ctx.save_for_backward(d, n, delta)

        return delta

    @staticmethod
    def backward(ctx, grad_output):
        """ Performs the backward pass of the Depth function.
        Args:
            ctx (Pytorch Autograd Context): pytorch autograd context
            grad_output (tensor): gradient outputs
        """
        # import pdb
        di, nj, deltai = ctx.saved_tensors
        # pdb.set_trace()
        # sdf of first intersection, first_intersection, in-direction, normal, delta
        return -grad_output / (nj * di).sum(-1), \
               -grad_output.unsqueeze(-1) * nj / (nj * di).sum(-1, keepdim=True), \
               (-grad_output * deltai).unsqueeze(-1) * nj / (nj * di).sum(-1, keepdim=True), \
               None, None
