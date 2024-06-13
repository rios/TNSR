"""
Transparent Neural Surface Refinement implementation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, cast
from torchtyping import TensorType

import numpy as np
import torch
from torch.nn import Parameter
from jaxtyping import Float
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import interlevel_loss
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.utils import colormaps
from tnsr.tnsr_neus import NeuSNSRModel, NeuSNSRModelConfig
from tnsr.tnsr_sampler import TraceModule, ProposalNetworkSampler


@dataclass
class NeuSNSRFactoModelConfig(NeuSNSRModelConfig):
    """NeusFacto Model Config"""

    _target: Type = field(default_factory=lambda: NeuSNSRFactoModel)
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for the proposal network."""
    num_neus_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    smooth_loss_multi: float = 0.005
    """smoothness loss on surface points in unisurf"""
    refinement: bool = False
    """Whether to refine the surface."""
    eta: float = 1.45
    """The refractive index of object"""
    cuboid_coordinates: Tuple[float, float, float, float, float, float] = (
        -0.15, -0.15, -0.15, 0.15, 0.15, 0.15)  # synthetic
    # (-0.14, -0.11, -0.06, 0.10, 0.13, 0.15)  # ball
    # (-0.15, -0.15, -0.15, 0.20, 0.16, 0.40)  # glass
    """The region of interest (object region)"""


def get_alphas(deltas: Float[Tensor, "*batch num_samples 1"],
               densities: Float[Tensor, "*batch num_samples 1"]) -> Float[Tensor, "*batch num_samples 1"]:

    """Return weights based on predicted densities

    Args:
        densities: Predicted densities for samples along ray

    Returns:
        Weights for each sample
    """

    delta_density = deltas * densities
    alphas = 1 - torch.exp(-delta_density)

    return alphas


class NeuSNSRFactoModel(NeuSNSRModel):
    """NeuSFactoModel extends NeuSModel for a more efficient sampling strategy.

    The model improves the rendering speed and quality by incorporating a learning-based
    proposal distribution to guide the sampling process.(similar to mipnerf-360)

    Args:
        config: NeuS configuration to instantiate model
    """

    config: NeuSNSRFactoModelConfig

    def populate_modules(self):
        """Instantiate modules and fields, including proposal networks."""
        super().populate_modules()

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb, spatial_distortion=self.scene_contraction, **prop_net_args
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=self.scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # update proposal network every iterations
        def update_schedule(_):
            return -1

        initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_neus_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        self.TraceModule = TraceModule(eta=self.config.eta)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return a dictionary with the parameters of the proposal networks."""
        param_groups = {}

        if self.config.refinement:
            clin_group = []
            other_group = []
            for module_name, module in self.field.named_modules():
                # Check if the module name contains "clin"
                if module_name == '':
                    continue
                if 'clin' in module_name:
                    clin_group.extend(module.parameters())
                else:
                    other_group.extend(module.parameters())

            param_groups["fields_color"] = clin_group
            param_groups["fields_sdf"] = other_group

            param_groups["field_background"] = (
                [self.field_background]
                if isinstance(self.field_background, Parameter)
                else list(self.field_background.parameters())
            )
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
            return param_groups
        else:
            param_groups["fields"] = list(self.field.parameters())
            param_groups["field_background"] = (
                [self.field_background]
                if isinstance(self.field_background, Parameter)
                else list(self.field_background.parameters())
            )
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step: int):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict[str, Any]:
        """Sample rays using proposal networks and compute the corresponding field outputs."""
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        if self.config.smooth_loss_multi > 0:
            surface_points = self.proposal_sampler.find_surface_points(ray_samples, sdf_fn=self.field.get_sdf)
        else:
            surface_points = []

        if self.config.refinement:
            ray_bundle_ref, selected_indices, ray_bundle_refl, attenuate, first_surface_points, second_surface_points = self.TraceModule(
                ray_bundle,
                sdf_pts_fn=self.field.get_point_sdf,
                cuboid_coordinates=self.config.cuboid_coordinates
            )
            if selected_indices != []:
                surface_points = torch.cat([first_surface_points, second_surface_points], dim=0)
        else:
            selected_indices = []

        field_outputs = self.field(ray_samples, return_alphas=True)

        if self.config.background_model != "none":
            field_outputs = self.forward_background_field_and_merge(ray_samples, field_outputs)

        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            # "bg_transmittance": bg_transmittance,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list,
            "surface_points": surface_points,
        }
        if selected_indices != []:
            # refraction
            out_ray_samples, _, _ = self.proposal_sampler(ray_bundle_ref[selected_indices],
                                                          density_fns=self.density_fns)
            out_field_outputs = self.field(out_ray_samples, return_alphas=True)

            if self.config.background_model != "none":
                out_field_outputs = self.forward_background_field_and_merge(out_ray_samples, out_field_outputs)

            out_weights = ray_samples.get_weights_and_transmittance_from_alphas(out_field_outputs[FieldHeadNames.ALPHA],
                                                                                weights_only=True)
            out_rgb = self.renderer_rgb(rgb=out_field_outputs[FieldHeadNames.RGB], weights=out_weights)
            samples_and_field_outputs.update({"out_rgb": out_rgb,
                                              "selected_indices": selected_indices}
                                             )

            # reflection
            refl_ray_samples, _, _ = self.proposal_sampler(ray_bundle_refl[selected_indices],
                                                           density_fns=self.density_fns)
            refl_field_outputs = self.field(refl_ray_samples, return_alphas=True)

            if self.config.background_model != "none":
                refl_field_outputs = self.forward_background_field_and_merge(refl_ray_samples, refl_field_outputs)

            refl_weights = ray_samples.get_weights_and_transmittance_from_alphas(
                refl_field_outputs[FieldHeadNames.ALPHA], weights_only=True)
            refl_rgb = self.renderer_rgb(rgb=refl_field_outputs[FieldHeadNames.RGB], weights=refl_weights)
            samples_and_field_outputs.update({"refl_rgb": refl_rgb,
                                              "attenuate": attenuate}
                                             )
        return samples_and_field_outputs

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        assert (
                ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata
        ), "directions_norm is required in ray_bundle.metadata"

        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        # shortcuts
        field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
            Dict[FieldHeadNames, torch.Tensor], samples_and_field_outputs["field_outputs"]
        )
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.metadata["directions_norm"]

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # background model
        if self.config.background_model != "none" and "bg_transmittance" in samples_and_field_outputs:
            bg_transmittance = samples_and_field_outputs["bg_transmittance"]
            assert isinstance(self.field_background, torch.nn.Module), "field_background should be a module"
            assert ray_bundle.fars is not None, "fars is required in ray_bundle"
            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            assert ray_bundle.fars is not None
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            assert not isinstance(self.field_background, Parameter)
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
            depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
            accumulation_bg = self.renderer_accumulation(weights=weights_bg)

            # merge background color to foregound color
            rgb = rgb + bg_transmittance * rgb_bg

            bg_outputs = {
                "bg_rgb": rgb_bg,
                "bg_accumulation": accumulation_bg,
                "bg_depth": depth_bg,
                "bg_weights": weights_bg,
            }
        else:
            bg_outputs = {}

        if "out_rgb" in samples_and_field_outputs.keys():
            selected_indices = samples_and_field_outputs["selected_indices"]
            attenuate = samples_and_field_outputs["attenuate"]
            rgb_matt = attenuate * samples_and_field_outputs["refl_rgb"] + (1 - attenuate) * samples_and_field_outputs[
                "out_rgb"]
            # Combine reflected and refracted components to final RGB.
            rgb_matt = torch.clamp(rgb_matt, 0.0, 1.0)
            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb_matt = rgb_matt * (1 + 2 * self.field.config.rgb_padding) - self.field.config.rgb_padding
            rgb[selected_indices] = rgb_matt

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            # used to scale z_vals for free space and sdf loss
            "directions_norm": ray_bundle.metadata["directions_norm"],
        }
        outputs.update(bg_outputs)

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)

        if "weights_list" in samples_and_field_outputs:
            weights_list = cast(List[torch.Tensor], samples_and_field_outputs["weights_list"])
            ray_samples_list = cast(List[torch.Tensor], samples_and_field_outputs["ray_samples_list"])

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs

    def forward_background_field_and_merge(self, ray_samples: RaySamples, field_outputs: Dict) -> Dict:
        """_summary_

        Args:
            ray_samples (RaySamples): _description_
            field_outputs (Dict): _description_
        """

        inside_sphere_mask = self.get_foreground_mask(ray_samples, roi=1.0)
        # TODO only forward the points that are outside the sphere if there is a background model

        field_outputs_bg = self.field_background(ray_samples)
        field_outputs_bg[FieldHeadNames.ALPHA] = get_alphas(ray_samples.deltas,
                                                                 field_outputs_bg[FieldHeadNames.DENSITY])

        field_outputs[FieldHeadNames.ALPHA] = (
                field_outputs[FieldHeadNames.ALPHA] * inside_sphere_mask
                + (1.0 - inside_sphere_mask) * field_outputs_bg[FieldHeadNames.ALPHA]
        )
        field_outputs[FieldHeadNames.RGB] = (
                field_outputs[FieldHeadNames.RGB] * inside_sphere_mask
                + (1.0 - inside_sphere_mask) * field_outputs_bg[FieldHeadNames.RGB]
        )
        # TODO make everything outside the sphere to be 0
        return field_outputs

    def get_foreground_mask(self, ray_samples: RaySamples, roi=1.0) -> TensorType:
        """_summary_

        Args:
            ray_samples (RaySamples): _description_
        """
        # TODO support multiple foreground type: box and sphere
        inside_sphere_mask = (ray_samples.frustums.get_start_positions().norm(dim=-1, keepdim=True) < roi).float()
        return inside_sphere_mask

    def get_loss_dict(
            self, outputs: Dict[str, Any], batch: Dict[str, Any], metrics_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute the loss dictionary, including interlevel loss for proposal networks."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

        if self.training and self.config.smooth_loss_multi > 0.0:
            surface_points = outputs["surface_points"]

            if surface_points != []:
                surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01
                pp = torch.cat([surface_points, surface_points_neig], dim=0)
                surface_grad = self.field.gradient(pp)
                surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)

                N = surface_points_normal.shape[0] // 2

                diff_norm = torch.norm(surface_points_normal[:N] - surface_points_normal[N:], dim=-1)
                loss_dict["normal_smoothness_loss"] = torch.mean(diff_norm) * self.config.smooth_loss_multi

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute image metrics and images, including the proposal depth for each iteration."""
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
