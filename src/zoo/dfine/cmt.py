"""
Conservative Moment Transport (CMT) components for CMT-FINE-MR.

The implementation is deliberately isolated from the official D-FINE backbone,
encoder, postprocessor, and evaluator.  CMT produces a non-negative evidence
measure, transports its low-order spatial moments, reads query-local moments,
and applies a vectorized soft KL projection to D-FINE's four edge distributions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register


__all__ = ["CMTMomentRefiner"]


class EvidenceEncoder(nn.Module):
    """Narrow, stride-one encoder that emits a non-negative localization measure."""

    def __init__(self, in_channels=3, hidden_channels=16, activation="softplus"):
        super().__init__()
        if activation not in ("softplus", "relu"):
            raise ValueError(f"Unsupported evidence activation: {activation}")
        self.activation = activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.SiLU(inplace=True),
        )
        self.logit = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.normal_(self.logit.weight, std=1e-3)
        nn.init.constant_(self.logit.bias, -4.0 if activation == "softplus" else 0.1)

    def forward(self, images):
        logits = self.logit(self.stem(images))
        if self.activation == "softplus":
            evidence = F.softplus(logits)
        else:
            evidence = F.relu(logits)
        return logits, evidence


def _moment_kernels(stride):
    coords = torch.arange(stride, dtype=torch.float32) + 0.5 - stride / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack(
        [torch.ones_like(xx), xx, yy, xx.square(), xx * yy, yy.square()], dim=0
    ).unsqueeze(1)


class MomentPyramid(nn.Module):
    """Encode stride-four moments and merge them with exact origin transport."""

    def __init__(self, base_stride=4, num_levels=4):
        super().__init__()
        if base_stride < 1 or num_levels < 1:
            raise ValueError("base_stride and num_levels must be positive")
        self.base_stride = int(base_stride)
        self.num_levels = int(num_levels)
        self.register_buffer("kernels", _moment_kernels(self.base_stride), persistent=False)

    @staticmethod
    def _merge_four(state, child_stride):
        b, channels, h, w = state.shape
        if channels != 6:
            raise ValueError(f"Expected six moment channels, got {channels}")
        if h % 2 or w % 2:
            state = F.pad(state, (0, w % 2, 0, h % 2))
            h, w = state.shape[-2:]

        child = state.reshape(b, 6, h // 2, 2, w // 2, 2).permute(0, 2, 4, 3, 5, 1)
        m, px, py, qxx, qxy, qyy = child.unbind(-1)
        half = float(child_stride) / 2.0
        dx = state.new_tensor([[-half, half], [-half, half]])
        dy = state.new_tensor([[-half, -half], [half, half]])
        reduce_dims = (-2, -1)

        parent_m = m.sum(reduce_dims)
        parent_px = (px + dx * m).sum(reduce_dims)
        parent_py = (py + dy * m).sum(reduce_dims)
        parent_qxx = (qxx + 2.0 * dx * px + dx.square() * m).sum(reduce_dims)
        parent_qxy = (qxy + dx * py + dy * px + dx * dy * m).sum(reduce_dims)
        parent_qyy = (qyy + 2.0 * dy * py + dy.square() * m).sum(reduce_dims)
        return torch.stack(
            [parent_m, parent_px, parent_py, parent_qxx, parent_qxy, parent_qyy], dim=1
        )

    def forward(self, evidence):
        original_size = tuple(evidence.shape[-2:])
        divisor = self.base_stride * (2 ** (self.num_levels - 1))
        pad_h = (-original_size[0]) % divisor
        pad_w = (-original_size[1]) % divisor
        if pad_h or pad_w:
            evidence = F.pad(evidence, (0, pad_w, 0, pad_h))

        evidence_fp32 = evidence.float()
        state = F.conv2d(evidence_fp32, self.kernels.to(evidence_fp32), stride=self.base_stride)
        levels = {self.base_stride: state}
        stride = self.base_stride
        for _ in range(1, self.num_levels):
            state = self._merge_four(state, stride)
            stride *= 2
            levels[stride] = state
        return {"levels": levels, "image_size": original_size, "padded_size": evidence.shape[-2:]}


class MomentReader(nn.Module):
    """O(1)-per-window rectangular moment reads using integral moment maps."""

    def __init__(self, base_stride=4, support_scales=(0.75, 1.5, 3.0), min_radius_px=4.0, eps=1e-6):
        super().__init__()
        if not support_scales or any(float(v) <= 0 for v in support_scales):
            raise ValueError("support_scales must contain positive values")
        self.base_stride = int(base_stride)
        self.support_scales = tuple(float(v) for v in support_scales)
        self.min_radius_px = float(min_radius_px)
        self.eps = float(eps)
        self.descriptor_dim = 8 * len(self.support_scales) + 1

    @staticmethod
    def _integral_image(values):
        return F.pad(values.cumsum(-1).cumsum(-2), (1, 0, 1, 0))

    @staticmethod
    def _gather(integral, x, y):
        b, channels, h, w = integral.shape
        index = (y * w + x).unsqueeze(1).expand(-1, channels, -1)
        return integral.flatten(2).gather(2, index)

    def _rectangle_sum(self, integral, x0, y0, x1, y1):
        return (
            self._gather(integral, x1, y1)
            - self._gather(integral, x0, y1)
            - self._gather(integral, x1, y0)
            + self._gather(integral, x0, y0)
        ).transpose(1, 2)

    def _global_raw_moments(self, state, moment_order):
        _, _, h, w = state.shape
        dtype, device = state.dtype, state.device
        oy = (torch.arange(h, dtype=dtype, device=device) + 0.5) * self.base_stride
        ox = (torch.arange(w, dtype=dtype, device=device) + 0.5) * self.base_stride
        yy, xx = torch.meshgrid(oy, ox, indexing="ij")
        xx, yy = xx.unsqueeze(0), yy.unsqueeze(0)

        m = state[:, 0]
        if moment_order == 0:
            px = py = torch.zeros_like(m)
        else:
            px, py = state[:, 1], state[:, 2]
        gx = px + xx * m
        gy = py + yy * m

        if moment_order >= 2:
            qxx = state[:, 3] + 2.0 * xx * px + xx.square() * m
            qxy = state[:, 4] + xx * py + yy * px + xx * yy * m
            qyy = state[:, 5] + 2.0 * yy * py + yy.square() * m
        else:
            qxx = qxy = qyy = torch.zeros_like(m)
        return torch.stack([m, gx, gy, qxx, qxy, qyy], dim=1)

    def prepare(self, pyramid, moment_order):
        """Cache the invariant integral moment map once per input batch."""
        state = pyramid["levels"][self.base_stride]
        pyramid["reader_moment_order"] = int(moment_order)
        pyramid["reader_integral"] = self._integral_image(
            self._global_raw_moments(state, moment_order)
        )
        return pyramid

    def forward(self, pyramid, reference_boxes, moment_order=2):
        if moment_order not in (0, 1, 2):
            raise ValueError(f"moment_order must be 0, 1, or 2, got {moment_order}")
        state = pyramid["levels"][self.base_stride]
        image_h, image_w = pyramid["image_size"]
        b, _, grid_h, grid_w = state.shape
        if reference_boxes.shape[0] != b or reference_boxes.shape[-1] != 4:
            raise ValueError("reference_boxes must have shape [B, Q, 4]")

        if pyramid.get("reader_moment_order") == moment_order and "reader_integral" in pyramid:
            integral = pyramid["reader_integral"]
        else:
            integral = self._integral_image(self._global_raw_moments(state, moment_order))
        ref_x = reference_boxes[..., 0].float() * float(image_w)
        ref_y = reference_boxes[..., 1].float() * float(image_h)
        box_w = reference_boxes[..., 2].float() * float(image_w)
        box_h = reference_boxes[..., 3].float() * float(image_h)
        reference_px = torch.stack([ref_x, ref_y], dim=-1)

        centers, descriptors, valid_scales, masses = [], [], [], []
        radii = []
        for scale in self.support_scales:
            radius_x = torch.clamp(0.5 * scale * box_w, min=self.min_radius_px)
            radius_y = torch.clamp(0.5 * scale * box_h, min=self.min_radius_px)
            radii.append(torch.stack([radius_x, radius_y], dim=-1))

            x0 = torch.floor((ref_x - radius_x) / self.base_stride).long().clamp(0, grid_w)
            y0 = torch.floor((ref_y - radius_y) / self.base_stride).long().clamp(0, grid_h)
            x1 = torch.ceil((ref_x + radius_x) / self.base_stride).long().clamp(0, grid_w)
            y1 = torch.ceil((ref_y + radius_y) / self.base_stride).long().clamp(0, grid_h)
            x1 = torch.maximum(x1, (x0 + 1).clamp(max=grid_w))
            y1 = torch.maximum(y1, (y0 + 1).clamp(max=grid_h))

            sums = self._rectangle_sum(integral, x0, y0, x1, y1)
            mass = sums[..., 0]
            valid = mass > self.eps
            safe_mass = mass.clamp_min(self.eps)
            center = sums[..., 1:3] / safe_mass.unsqueeze(-1)
            center = torch.where(valid.unsqueeze(-1), center, reference_px)

            if moment_order >= 2:
                exx = sums[..., 3] / safe_mass
                exy = sums[..., 4] / safe_mass
                eyy = sums[..., 5] / safe_mass
                cov_xx = (exx - center[..., 0].square()).clamp_min(0.0)
                cov_xy = exy - center[..., 0] * center[..., 1]
                cov_yy = (eyy - center[..., 1].square()).clamp_min(0.0)
            else:
                cov_xx = cov_xy = cov_yy = torch.zeros_like(mass)

            left = torch.maximum(ref_x - radius_x, ref_x.new_zeros(()))
            right = torch.minimum(ref_x + radius_x, ref_x.new_tensor(float(image_w)))
            top = torch.maximum(ref_y - radius_y, ref_y.new_zeros(()))
            bottom = torch.minimum(ref_y + radius_y, ref_y.new_tensor(float(image_h)))
            valid_fraction = ((right - left).clamp_min(0) / (2.0 * radius_x)).clamp(0, 1)
            valid_fraction *= ((bottom - top).clamp_min(0) / (2.0 * radius_y)).clamp(0, 1)
            area = (4.0 * radius_x * radius_y).clamp_min(1.0)
            offset = center - reference_px
            descriptor = torch.stack(
                [
                    torch.log1p(mass / area),
                    offset[..., 0] / radius_x,
                    offset[..., 1] / radius_y,
                    cov_xx / radius_x.square(),
                    cov_xy / (radius_x * radius_y),
                    cov_yy / radius_y.square(),
                    valid.to(mass.dtype),
                    valid_fraction,
                ],
                dim=-1,
            )
            centers.append(center)
            descriptors.append(descriptor)
            valid_scales.append(valid)
            masses.append(mass)

        centers = torch.stack(centers, dim=2)
        valid_scales = torch.stack(valid_scales, dim=2)
        masses = torch.stack(masses, dim=2)
        radii = torch.stack(radii, dim=2)
        if centers.shape[2] > 1:
            pairwise = centers.unsqueeze(3) - centers.unsqueeze(2)
            disagreement = pairwise.square().sum(-1).clamp_min(1e-12).sqrt().amax(dim=(2, 3))
            disagreement = disagreement / float(max(image_h, image_w))
        else:
            disagreement = centers.new_zeros(centers.shape[:2])
        descriptor = torch.cat(descriptors + [disagreement.unsqueeze(-1)], dim=-1)
        return {
            "centers_px": centers,
            "descriptor": descriptor,
            "valid_scales": valid_scales,
            "masses": masses,
            "radii_px": radii,
            "reference_px": reference_px,
            "image_size": (image_h, image_w),
        }


class MomentDistributionProjector(nn.Module):
    """Vectorized soft KL projection of D-FINE edge-distribution logits."""

    def __init__(self, reg_max=32, strength=1.0, solver_steps=4, max_dual=20.0):
        super().__init__()
        self.reg_max = int(reg_max)
        self.strength = float(strength)
        self.solver_steps = int(solver_steps)
        self.max_dual = float(max_dual)
        if self.strength < 0.0:
            raise ValueError("projection strength must be non-negative")
        if self.solver_steps < 1:
            raise ValueError("solver_steps must be positive")
        if self.max_dual <= 0.0:
            raise ValueError("max_dual must be positive")

    @staticmethod
    def _mean_and_var(logits, support):
        prob = logits.softmax(dim=-1)
        mean = (prob * support).sum(dim=-1)
        variance = (prob * support.square()).sum(dim=-1) - mean.square()
        return mean, variance.clamp_min(0.0)

    def _solve_axis(self, negative_logits, positive_logits, support, c0, target, scale, rho):
        dual = torch.zeros_like(c0)
        for _ in range(self.solver_steps):
            neg_mean, neg_var = self._mean_and_var(
                negative_logits - dual.unsqueeze(-1) * scale.unsqueeze(-1) * support, support
            )
            pos_mean, pos_var = self._mean_and_var(
                positive_logits + dual.unsqueeze(-1) * scale.unsqueeze(-1) * support, support
            )
            center = c0 + scale * (pos_mean - neg_mean)
            residual = dual - rho * (target - center)
            derivative = 1.0 + rho * scale.square() * (pos_var + neg_var)
            dual = (dual - residual / derivative.clamp_min(1e-6)).clamp(
                -self.max_dual, self.max_dual
            )
        return dual

    def forward(
        self, logits, reference_boxes, moment_center_px, gate, support, image_size, reg_scale
    ):
        if self.strength == 0.0:
            return logits
        bins = self.reg_max + 1
        if logits.shape[-1] != 4 * bins:
            raise ValueError(f"Expected {4 * bins} corner logits, got {logits.shape[-1]}")
        image_h, image_w = image_size
        original_dtype = logits.dtype
        corner_logits = logits.float().reshape(*logits.shape[:-1], 4, bins)
        support = support.float().reshape(1, 1, bins)
        reference_boxes = reference_boxes.float()
        moment_center_px = moment_center_px.float()
        gate = gate.float().squeeze(-1)

        reg_scale = reg_scale.detach().float().abs().clamp_min(1e-6)
        width_px = reference_boxes[..., 2] * float(image_w)
        height_px = reference_boxes[..., 3] * float(image_h)
        cx0 = reference_boxes[..., 0] * float(image_w)
        cy0 = reference_boxes[..., 1] * float(image_h)
        ax = width_px / (2.0 * reg_scale)
        ay = height_px / (2.0 * reg_scale)
        rho_x = gate * self.strength / width_px.clamp_min(4.0).square()
        rho_y = gate * self.strength / height_px.clamp_min(4.0).square()

        dual_x = self._solve_axis(
            corner_logits[..., 0, :],
            corner_logits[..., 2, :],
            support,
            cx0,
            moment_center_px[..., 0],
            ax,
            rho_x,
        )
        dual_y = self._solve_axis(
            corner_logits[..., 1, :],
            corner_logits[..., 3, :],
            support,
            cy0,
            moment_center_px[..., 1],
            ay,
            rho_y,
        )
        corner_logits = corner_logits.clone()
        corner_logits[..., 0, :] -= dual_x.unsqueeze(-1) * ax.unsqueeze(-1) * support
        corner_logits[..., 2, :] += dual_x.unsqueeze(-1) * ax.unsqueeze(-1) * support
        corner_logits[..., 1, :] -= dual_y.unsqueeze(-1) * ay.unsqueeze(-1) * support
        corner_logits[..., 3, :] += dual_y.unsqueeze(-1) * ay.unsqueeze(-1) * support
        return corner_logits.flatten(-2).to(original_dtype)


@register()
class CMTMomentRefiner(nn.Module):
    """Optional CMT-FINE-MR branch with explicit ablation modes."""

    _MODES = {"baseline", "evidence_only", "mass_only", "first_order", "full"}

    def __init__(
        self,
        enabled=True,
        mode="full",
        hidden_dim=256,
        evidence_channels=16,
        evidence_activation="softplus",
        base_stride=4,
        num_moment_levels=4,
        support_scales=(0.75, 1.5, 3.0),
        min_radius_px=4.0,
        reg_max=32,
        projection_strength=1.0,
        solver_steps=4,
        learned_gate=True,
        fixed_gate=0.5,
        class_residual=True,
        class_residual_init=0.01,
        refine_layers="all",
    ):
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {sorted(self._MODES)}, got {mode}")
        if refine_layers not in ("all", "last") and not isinstance(refine_layers, (list, tuple)):
            raise ValueError("refine_layers must be 'all', 'last', or a list of layer indices")
        if isinstance(refine_layers, (list, tuple)) and (
            not refine_layers
            or any(not isinstance(index, int) or index < 0 for index in refine_layers)
        ):
            raise ValueError("explicit refine_layers must contain non-negative layer indices")
        if not 0.0 <= float(fixed_gate) <= 1.0:
            raise ValueError("fixed_gate must be in [0, 1]")
        self.enabled = bool(enabled)
        self.mode = mode
        self.hidden_dim = int(hidden_dim)
        self.learned_gate = bool(learned_gate)
        self.fixed_gate = float(fixed_gate)
        self.class_residual = bool(class_residual)
        self.refine_layers = refine_layers
        self.evidence = EvidenceEncoder(3, int(evidence_channels), evidence_activation)
        self.pyramid = MomentPyramid(base_stride, num_moment_levels)
        self.reader = MomentReader(base_stride, support_scales, min_radius_px)
        descriptor_dim = self.reader.descriptor_dim
        gate_hidden = max(32, self.hidden_dim // 4)
        self.scale_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + descriptor_dim),
            nn.Linear(self.hidden_dim + descriptor_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, len(support_scales)),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + descriptor_dim),
            nn.Linear(self.hidden_dim + descriptor_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.class_descriptor_head = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, self.hidden_dim),
        )
        self.class_residual_scale = nn.Parameter(
            torch.tensor(float(class_residual_init), dtype=torch.float32)
        )
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        initial_gate = min(max(self.fixed_gate, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.gate_head[-1].bias, math.log(initial_gate / (1.0 - initial_gate)))
        self.projector = MomentDistributionProjector(
            reg_max=reg_max, strength=projection_strength, solver_steps=solver_steps
        )
        self._freeze_disabled_paths()

    def _freeze_disabled_paths(self):
        """Keep official DDP(find_unused_parameters=False) safe for ablation configs."""
        if not self.active:
            self.requires_grad_(False)
            return
        if not self.learned_gate:
            self.gate_head.requires_grad_(False)
        if not self.class_residual:
            self.class_descriptor_head.requires_grad_(False)
            self.class_residual_scale.requires_grad_(False)

    @property
    def active(self):
        return self.enabled and self.mode != "baseline"

    def prepare(self, images):
        if not self.active:
            return None
        evidence_logits, evidence = self.evidence(images)
        pyramid = self.pyramid(evidence)
        order = {"evidence_only": 2, "mass_only": 0, "first_order": 1, "full": 2}[self.mode]
        pyramid = self.reader.prepare(pyramid, order)
        pyramid["evidence_logits"] = evidence_logits
        return pyramid

    def _layer_enabled(self, layer_index, final_index):
        if self.refine_layers == "all":
            return True
        if self.refine_layers == "last":
            return layer_index == final_index
        return layer_index in self.refine_layers

    def refine(
        self,
        logits,
        query,
        reference_boxes,
        support,
        reg_scale,
        state,
        layer_index,
        final_index,
    ):
        if not self.active or state is None or not self._layer_enabled(layer_index, final_index):
            return logits, None
        if query.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"CMT hidden_dim={self.hidden_dim} does not match decoder query dim={query.shape[-1]}"
            )
        order = {"evidence_only": 2, "mass_only": 0, "first_order": 1, "full": 2}[self.mode]
        readout = self.reader(state, reference_boxes, moment_order=order)
        descriptor = readout["descriptor"].to(query.dtype)
        joint = torch.cat([query, descriptor], dim=-1)
        scale_logits = self.scale_head(joint).float()
        valid_scales = readout["valid_scales"]
        scale_logits = scale_logits.masked_fill(~valid_scales, -1e4)
        scale_weights = scale_logits.softmax(dim=-1)
        moment_center_px = (scale_weights.unsqueeze(-1) * readout["centers_px"]).sum(dim=2)
        valid = valid_scales.any(dim=-1)

        if self.learned_gate:
            gate_logits = self.gate_head(joint).float()
            gate = gate_logits.sigmoid()
        else:
            gate_logits = query.new_full((*query.shape[:2], 1), float("nan"), dtype=torch.float32)
            gate = query.new_full((*query.shape[:2], 1), self.fixed_gate, dtype=torch.float32)
        gate = gate * valid.unsqueeze(-1).to(gate.dtype)

        if self.class_residual:
            class_delta = self.class_descriptor_head(descriptor).to(query.dtype)
            classification_query = query + self.class_residual_scale.to(query.dtype) * class_delta
        else:
            classification_query = query

        if self.mode == "evidence_only":
            refined = logits
        else:
            refined = self.projector(
                logits,
                reference_boxes,
                moment_center_px,
                gate,
                support,
                readout["image_size"],
                reg_scale,
            )
        image_h, image_w = readout["image_size"]
        normalizer = moment_center_px.new_tensor([float(image_w), float(image_h)])
        diagnostics = {
            "moment_center": moment_center_px / normalizer,
            "semantic_center": reference_boxes[..., :2],
            "gate": gate,
            "gate_logits": gate_logits,
            "valid": valid,
            "mass": (scale_weights * readout["masses"]).sum(dim=-1),
            "scale_weights": scale_weights,
            "image_size": moment_center_px.new_tensor([image_h, image_w]),
            "classification_query": classification_query,
        }
        return refined, diagnostics
