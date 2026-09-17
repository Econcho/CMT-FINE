"""
Conservative Moment Transport (CMT) components for CMT-FINE-MR.

The implementation is deliberately isolated from the official D-FINE backbone,
encoder, postprocessor, and evaluator.  CMT produces a non-negative evidence
measure, transports its low-order spatial moments, reads query-local moments,
and applies a vectorized soft KL projection to D-FINE's four edge distributions.
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register


__all__ = ["CMTMomentRefiner"]


def _autocast_disabled(device):
    """Disable CUDA/CPU autocast for numerically sensitive moment operations."""
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


class EvidenceEncoder(nn.Module):
    """Fuse image detail and encoder semantics into shared non-negative evidence bases."""

    def __init__(
        self,
        in_channels=3,
        semantic_channels=256,
        hidden_channels=24,
        num_bases=8,
        activation="softplus",
    ):
        super().__init__()
        if activation not in ("softplus", "relu"):
            raise ValueError(f"Unsupported evidence activation: {activation}")
        self.activation = activation
        self.num_bases = int(num_bases)
        if self.num_bases < 1:
            raise ValueError("num_bases must be positive")
        self.detail = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.semantic = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.logit = nn.Conv2d(hidden_channels, self.num_bases, 1)
        nn.init.normal_(self.logit.weight, std=1e-3)
        nn.init.constant_(self.logit.bias, -4.0 if activation == "softplus" else 0.1)

    def forward(self, images, semantic_features=None):
        detail = self.detail(images)
        if semantic_features is None:
            semantic = torch.zeros_like(detail)
        else:
            semantic = self.semantic(semantic_features)
            semantic = F.interpolate(
                semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False
            )
        logits = self.logit(self.fuse(torch.cat([detail, semantic], dim=1)))
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
        # ``F.conv2d`` is autocast to FP16 even when its input was explicitly
        # converted to float. Keep all absolute and second-order moments in
        # FP32; otherwise sums such as x^2*m overflow on ordinary 640 inputs.
        with _autocast_disabled(evidence.device):
            original_size = tuple(evidence.shape[-2:])
            divisor = self.base_stride * (2 ** (self.num_levels - 1))
            pad_h = (-original_size[0]) % divisor
            pad_w = (-original_size[1]) % divisor
            if pad_h or pad_w:
                evidence = F.pad(evidence, (0, pad_w, 0, pad_h))

            evidence_fp32 = evidence.float()
            state = F.conv2d(
                evidence_fp32, self.kernels.to(evidence_fp32), stride=self.base_stride
            )
            levels = {self.base_stride: state}
            stride = self.base_stride
            for _ in range(1, self.num_levels):
                state = self._merge_four(state, stride)
                stride *= 2
                levels[stride] = state
            return {
                "levels": levels,
                "image_size": original_size,
                "padded_size": evidence.shape[-2:],
            }

    def from_base_evidence(self, evidence, image_size):
        """Build moments from K stride-``base_stride`` evidence bases.

        Each evidence value is treated as mass at the centre of its stride cell.
        The returned level tensors have shape ``[B, K, 6, H, W]``.
        """
        if evidence.ndim != 4:
            raise ValueError("base evidence must have shape [B, K, H, W]")
        with _autocast_disabled(evidence.device):
            evidence = evidence.float()
            b, k, h, w = evidence.shape
            zeros = torch.zeros_like(evidence)
            state = torch.stack([evidence, zeros, zeros, zeros, zeros, zeros], dim=2)
            levels = {self.base_stride: state}
            stride = self.base_stride
            for _ in range(1, self.num_levels):
                flat = state.reshape(b * k, 6, state.shape[-2], state.shape[-1])
                flat = self._merge_four(flat, stride)
                state = flat.reshape(b, k, 6, flat.shape[-2], flat.shape[-1])
                stride *= 2
                levels[stride] = state
            return {
                "levels": levels,
                "image_size": tuple(image_size),
                "padded_size": (h * self.base_stride, w * self.base_stride),
            }


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
        if integral.ndim == 4:
            b, channels, h, w = integral.shape
            index = (y * w + x).unsqueeze(1).expand(-1, channels, -1)
            return integral.flatten(2).gather(2, index)
        if integral.ndim != 5:
            raise ValueError("integral moments must be [B,6,H,W] or [B,K,6,H,W]")
        b, bases, channels, h, w = integral.shape
        index = (y * w + x).unsqueeze(1).unsqueeze(1).expand(-1, bases, channels, -1)
        return integral.flatten(3).gather(3, index)

    def _rectangle_sum(self, integral, x0, y0, x1, y1):
        sums = (
            self._gather(integral, x1, y1)
            - self._gather(integral, x0, y1)
            - self._gather(integral, x1, y0)
            + self._gather(integral, x0, y0)
        )
        return sums.transpose(1, 2) if sums.ndim == 3 else sums.permute(0, 3, 1, 2)

    def _global_raw_moments(self, state, moment_order):
        multi_basis = state.ndim == 5
        if multi_basis:
            b, bases, _, h, w = state.shape
            flat_state = state.reshape(b * bases, 6, h, w)
        else:
            flat_state = state
            b, bases = state.shape[0], 1
            _, _, h, w = state.shape
        dtype, device = flat_state.dtype, flat_state.device
        oy = (torch.arange(h, dtype=dtype, device=device) + 0.5) * self.base_stride
        ox = (torch.arange(w, dtype=dtype, device=device) + 0.5) * self.base_stride
        yy, xx = torch.meshgrid(oy, ox, indexing="ij")
        xx, yy = xx.unsqueeze(0), yy.unsqueeze(0)

        m = flat_state[:, 0]
        if moment_order == 0:
            px = py = torch.zeros_like(m)
        else:
            px, py = flat_state[:, 1], flat_state[:, 2]
        gx = px + xx * m
        gy = py + yy * m

        if moment_order >= 2:
            qxx = flat_state[:, 3] + 2.0 * xx * px + xx.square() * m
            qxy = flat_state[:, 4] + xx * py + yy * px + xx * yy * m
            qyy = flat_state[:, 5] + 2.0 * yy * py + yy.square() * m
        else:
            qxx = qxy = qyy = torch.zeros_like(m)
        result = torch.stack([m, gx, gy, qxx, qxy, qyy], dim=1)
        return result.reshape(b, bases, 6, h, w) if multi_basis else result

    def prepare(self, pyramid, moment_order):
        """Cache the invariant integral moment map once per input batch."""
        state = pyramid["levels"][self.base_stride]
        with _autocast_disabled(state.device):
            pyramid["reader_moment_order"] = int(moment_order)
            pyramid["reader_integral"] = self._integral_image(
                self._global_raw_moments(state, moment_order)
            )
        return pyramid

    def forward(self, pyramid, reference_boxes, moment_order=2, basis_weights=None):
        if moment_order not in (0, 1, 2):
            raise ValueError(f"moment_order must be 0, 1, or 2, got {moment_order}")
        state = pyramid["levels"][self.base_stride]
        image_h, image_w = pyramid["image_size"]
        if state.ndim == 5:
            b, bases, _, grid_h, grid_w = state.shape
            if basis_weights is None or basis_weights.shape != reference_boxes.shape[:2] + (bases,):
                raise ValueError("basis_weights must have shape [B, Q, K]")
        else:
            b, _, grid_h, grid_w = state.shape
            bases = 1
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
            if sums.ndim == 4:
                sums = (sums * basis_weights.unsqueeze(-1).float()).sum(dim=2)
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
        with _autocast_disabled(logits.device):
            return self._forward_fp32(
                logits,
                reference_boxes,
                moment_center_px,
                gate,
                support,
                image_size,
                reg_scale,
            )

    def _forward_fp32(
        self, logits, reference_boxes, moment_center_px, gate, support, image_size, reg_scale
    ):
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
    """Query-conditioned, quality-gated moment refinement for D-FINE."""

    _MODES = {"baseline", "evidence_only", "mass_only", "first_order", "full"}

    def __init__(
        self,
        enabled=True,
        mode="full",
        hidden_dim=256,
        semantic_channels=256,
        evidence_channels=24,
        evidence_bases=8,
        evidence_activation="softplus",
        base_stride=4,
        num_moment_levels=4,
        support_scales=(0.5, 1.0, 2.0),
        min_radius_px=2.0,
        reg_max=32,
        projection_strength=1.0,
        solver_steps=4,
        gain_levels=(0.0, 0.25, 0.5, 1.0),
        fixed_gain=None,
        tiny_side_px=32.0,
        size_temperature=0.5,
        min_size_budget=0.15,
        max_size_budget=1.0,
        refine_layers="last",
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
        if not gain_levels or any(float(v) < 0.0 or float(v) > 1.0 for v in gain_levels):
            raise ValueError("gain_levels must be non-empty and contained in [0, 1]")
        if tuple(sorted(float(v) for v in gain_levels)) != tuple(float(v) for v in gain_levels):
            raise ValueError("gain_levels must be sorted")
        if fixed_gain is not None and not 0.0 <= float(fixed_gain) <= 1.0:
            raise ValueError("fixed_gain must be in [0, 1]")
        if min_size_budget < 0.0 or max_size_budget < min_size_budget:
            raise ValueError("size budgets must satisfy 0 <= min <= max")
        self.enabled = bool(enabled)
        self.mode = mode
        self.hidden_dim = int(hidden_dim)
        self.evidence_bases = int(evidence_bases)
        self.fixed_gain = None if fixed_gain is None else float(fixed_gain)
        self.tiny_side_px = float(tiny_side_px)
        self.size_temperature = float(size_temperature)
        self.min_size_budget = float(min_size_budget)
        self.max_size_budget = float(max_size_budget)
        self.refine_layers = refine_layers
        self.evidence = EvidenceEncoder(
            in_channels=3,
            semantic_channels=int(semantic_channels),
            hidden_channels=int(evidence_channels),
            num_bases=self.evidence_bases,
            activation=evidence_activation,
        )
        self.pyramid = MomentPyramid(base_stride, num_moment_levels)
        self.reader = MomentReader(base_stride, support_scales, min_radius_px)
        descriptor_dim = self.reader.descriptor_dim
        decision_dim = self.hidden_dim + descriptor_dim + 2
        gate_hidden = max(32, self.hidden_dim // 4)
        self.basis_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, self.evidence_bases),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(decision_dim),
            nn.Linear(decision_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, len(support_scales)),
        )
        self.gain_head = nn.Sequential(
            nn.LayerNorm(decision_dim),
            nn.Linear(decision_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, len(gain_levels)),
        )
        self.register_buffer(
            "gain_levels", torch.tensor(tuple(float(v) for v in gain_levels)), persistent=True
        )
        nn.init.zeros_(self.basis_head[-1].weight)
        nn.init.zeros_(self.basis_head[-1].bias)
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)
        self.gain_head[-1].bias.data[0] = 2.0
        self.projector = MomentDistributionProjector(
            reg_max=reg_max, strength=projection_strength, solver_steps=solver_steps
        )
        self._freeze_disabled_paths()

    def _freeze_disabled_paths(self):
        """Keep official DDP(find_unused_parameters=False) safe for ablation configs."""
        if not self.active:
            self.requires_grad_(False)
            return
        if self.fixed_gain is not None:
            self.gain_head.requires_grad_(False)

    @property
    def active(self):
        return self.enabled and self.mode != "baseline"

    def prepare(self, images, semantic_features=None):
        if not self.active:
            return None
        evidence_logits, evidence = self.evidence(images, semantic_features)
        pyramid = self.pyramid.from_base_evidence(evidence, images.shape[-2:])
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
        current_boxes,
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
        basis_weights = self.basis_head(query).float().softmax(dim=-1)
        readout = self.reader(
            state,
            current_boxes.detach(),
            moment_order=order,
            basis_weights=basis_weights,
        )
        descriptor = readout["descriptor"].to(query.dtype)
        image_h, image_w = readout["image_size"]
        box_size_px = current_boxes[..., 2:4].detach().float() * current_boxes.new_tensor(
            [float(image_w), float(image_h)]
        )
        box_size_feature = torch.log1p(box_size_px.clamp_min(0.0) / self.tiny_side_px).to(
            query.dtype
        )
        joint = torch.cat([query, descriptor, box_size_feature], dim=-1)
        scale_logits = self.scale_head(joint).float()
        valid_scales = readout["valid_scales"]
        scale_logits = scale_logits.masked_fill(~valid_scales, -1e4)
        scale_weights = scale_logits.softmax(dim=-1)
        moment_center_px = (scale_weights.unsqueeze(-1) * readout["centers_px"]).sum(dim=2)
        valid = valid_scales.any(dim=-1)

        if self.fixed_gain is None:
            gain_logits = self.gain_head(joint).float()
            gain_probability = gain_logits.softmax(dim=-1)
            gain = (gain_probability * self.gain_levels.to(gain_probability)).sum(dim=-1, keepdim=True)
        else:
            gain_logits = query.new_full(
                (*query.shape[:2], self.gain_levels.numel()), float("nan"), dtype=torch.float32
            )
            gain = query.new_full((*query.shape[:2], 1), self.fixed_gain, dtype=torch.float32)
        gain = gain * valid.unsqueeze(-1).to(gain.dtype)

        side_px = (
            current_boxes[..., 2].detach().float().clamp_min(1e-6)
            * current_boxes[..., 3].detach().float().clamp_min(1e-6)
            * float(image_h * image_w)
        ).sqrt()
        tiny_priority = torch.sigmoid(
            (math.log(self.tiny_side_px) - side_px.log()) / max(self.size_temperature, 1e-3)
        )
        size_budget = self.min_size_budget + (
            self.max_size_budget - self.min_size_budget
        ) * tiny_priority
        size_budget = size_budget.unsqueeze(-1) * valid.unsqueeze(-1).to(size_budget.dtype)

        if self.mode == "evidence_only":
            candidate = logits
        else:
            candidate = self.projector(
                logits,
                reference_boxes,
                moment_center_px,
                size_budget,
                support,
                readout["image_size"],
                reg_scale,
            )
        if self.fixed_gain == 0.0:
            refined = logits
        else:
            bins = self.projector.reg_max + 1
            base_probability = logits.float().reshape(*logits.shape[:-1], 4, bins).softmax(-1)
            candidate_probability = candidate.float().reshape(*candidate.shape[:-1], 4, bins).softmax(-1)
            refined_probability = (
                (1.0 - gain.unsqueeze(-1)) * base_probability
                + gain.unsqueeze(-1) * candidate_probability
            ).clamp_min(1e-8)
            refined = refined_probability.log().flatten(-2).to(logits.dtype)

        base_boxes = self._decode_boxes(logits, reference_boxes, support, reg_scale)
        candidate_boxes = self._decode_boxes(candidate, reference_boxes, support, reg_scale)
        normalizer = moment_center_px.new_tensor([float(image_w), float(image_h)])
        diagnostics = {
            "moment_center": moment_center_px / normalizer,
            "semantic_center": current_boxes[..., :2],
            "base_boxes": base_boxes,
            "candidate_boxes": candidate_boxes,
            "gain": gain,
            "gain_logits": gain_logits,
            "gain_levels": self.gain_levels,
            "size_budget": size_budget,
            "basis_weights": basis_weights,
            "valid": valid,
            "mass": (scale_weights * readout["masses"]).sum(dim=-1),
            "scale_weights": scale_weights,
            "image_size": moment_center_px.new_tensor([image_h, image_w]),
            "classification_query": query,
        }
        return refined, diagnostics

    @staticmethod
    def _decode_boxes(logits, reference_boxes, support, reg_scale):
        bins = logits.shape[-1] // 4
        probability = logits.float().reshape(*logits.shape[:-1], 4, bins).softmax(-1)
        distance = (probability * support.float().reshape(1, 1, 1, bins)).sum(-1)
        scale = reg_scale.detach().float().abs().clamp_min(1e-6)
        x1 = reference_boxes[..., 0] - (0.5 * scale + distance[..., 0]) * (
            reference_boxes[..., 2] / scale
        )
        y1 = reference_boxes[..., 1] - (0.5 * scale + distance[..., 1]) * (
            reference_boxes[..., 3] / scale
        )
        x2 = reference_boxes[..., 0] + (0.5 * scale + distance[..., 2]) * (
            reference_boxes[..., 2] / scale
        )
        y2 = reference_boxes[..., 1] + (0.5 * scale + distance[..., 3]) * (
            reference_boxes[..., 3] / scale
        )
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
