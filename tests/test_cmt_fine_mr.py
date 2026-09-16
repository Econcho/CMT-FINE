import importlib.util
import pathlib
import sys
import types
import unittest

import torch
import torch.nn.functional as F


def _load_cmt_module():
    """Load the isolated CMT module without importing optional COCO training dependencies."""
    root = pathlib.Path(__file__).resolve().parents[1]
    for name, path in (
        ("src", root / "src"),
        ("src.zoo", root / "src" / "zoo"),
        ("src.zoo.dfine", root / "src" / "zoo" / "dfine"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules.setdefault(name, module)
    core = types.ModuleType("src.core")
    core.register = lambda *args, **kwargs: (lambda value: value)
    sys.modules["src.core"] = core
    path = root / "src" / "zoo" / "dfine" / "cmt.py"
    spec = importlib.util.spec_from_file_location("src.zoo.dfine.cmt", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmt = _load_cmt_module()


class MomentPyramidTest(unittest.TestCase):
    def test_hierarchical_merge_matches_direct_moments(self):
        torch.manual_seed(7)
        evidence = torch.rand(2, 1, 32, 32)
        pyramid = cmt.MomentPyramid(base_stride=4, num_levels=4)(evidence)
        direct = F.conv2d(evidence, cmt._moment_kernels(8), stride=8)
        torch.testing.assert_close(pyramid["levels"][8], direct, rtol=1e-5, atol=1e-4)

    def test_moment_pyramid_stays_fp32_under_autocast(self):
        evidence = torch.rand(1, 1, 64, 64)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            pyramid = cmt.MomentPyramid(base_stride=4, num_levels=4)(evidence)
        self.assertEqual(pyramid["levels"][4].dtype, torch.float32)
        self.assertEqual(pyramid["levels"][32].dtype, torch.float32)

    def test_first_order_reader_tracks_a_one_pixel_shift(self):
        evidence0 = torch.zeros(1, 1, 32, 32)
        evidence1 = torch.zeros_like(evidence0)
        evidence0[0, 0, 15, 15] = 1.0
        evidence1[0, 0, 15, 16] = 1.0
        pyramid = cmt.MomentPyramid(base_stride=4, num_levels=4)
        reader = cmt.MomentReader(base_stride=4, support_scales=(3.0,), min_radius_px=8.0)
        reference0 = torch.tensor([[[15.5 / 32, 15.5 / 32, 0.25, 0.25]]])
        reference1 = reference0.clone()
        reference1[..., 0] += 1.0 / 32
        center0 = reader(pyramid(evidence0), reference0, moment_order=1)["centers_px"][0, 0, 0]
        center1 = reader(pyramid(evidence1), reference1, moment_order=1)["centers_px"][0, 0, 0]
        torch.testing.assert_close(center1 - center0, torch.tensor([1.0, 0.0]))


class AblationConfigurationTest(unittest.TestCase):
    def test_disabled_paths_are_frozen_for_official_ddp_mode(self):
        baseline = cmt.CMTMomentRefiner(hidden_dim=16, reg_max=8, mode="baseline")
        self.assertFalse(any(parameter.requires_grad for parameter in baseline.parameters()))

        fixed = cmt.CMTMomentRefiner(
            hidden_dim=16,
            reg_max=8,
            learned_gate=False,
            class_residual=False,
        )
        self.assertFalse(any(parameter.requires_grad for parameter in fixed.gate_head.parameters()))
        self.assertFalse(
            any(parameter.requires_grad for parameter in fixed.class_descriptor_head.parameters())
        )
        self.assertFalse(fixed.class_residual_scale.requires_grad)
        self.assertTrue(any(parameter.requires_grad for parameter in fixed.evidence.parameters()))


class DistributionProjectionTest(unittest.TestCase):
    def test_zero_strength_is_bitwise_identity(self):
        logits = torch.randn(2, 5, 4 * 9)
        reference = torch.rand(2, 5, 4)
        center = torch.rand(2, 5, 2) * 64
        gate = torch.rand(2, 5, 1)
        support = torch.linspace(-2, 2, 9)
        projector = cmt.MomentDistributionProjector(reg_max=8, strength=0.0)
        actual = projector(logits, reference, center, gate, support, (64, 64), torch.tensor([4.0]))
        self.assertTrue(torch.equal(actual, logits))

    def test_projection_moves_distribution_center_toward_moment_center(self):
        support = torch.linspace(-2, 2, 9)
        logits = torch.zeros(1, 1, 4 * 9)
        reference = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        target_center = torch.tensor([[[380.0, 320.0]]])
        projector = cmt.MomentDistributionProjector(reg_max=8, strength=100.0, solver_steps=8)
        projected = projector(
            logits,
            reference,
            target_center,
            torch.ones(1, 1, 1),
            support,
            (640, 640),
            torch.tensor([4.0]),
        )
        probability = projected.reshape(1, 1, 4, 9).softmax(-1)
        mean = (probability * support).sum(-1)
        projected_x = 320.0 + 16.0 * (mean[..., 2] - mean[..., 0])
        self.assertGreater(projected_x.item(), 320.0)
        self.assertLess(abs(projected_x.item() - 380.0), 60.0)

    def test_full_refiner_has_finite_gradients(self):
        torch.manual_seed(11)
        module = cmt.CMTMomentRefiner(
            hidden_dim=16,
            evidence_channels=4,
            reg_max=8,
            projection_strength=5.0,
        )
        images = torch.rand(2, 3, 64, 64, requires_grad=True)
        state = module.prepare(images)
        query = torch.rand(2, 7, 16, requires_grad=True)
        reference = torch.tensor([0.5, 0.5, 0.25, 0.25]).reshape(1, 1, 4).repeat(2, 7, 1)
        logits = torch.zeros(2, 7, 4 * 9, requires_grad=True)
        refined, diagnostics = module.refine(
            logits,
            query,
            reference,
            torch.linspace(-2, 2, 9),
            torch.tensor([4.0]),
            state,
            layer_index=0,
            final_index=2,
        )
        loss = (
            refined.square().mean()
            + diagnostics["moment_center"].mean()
            + diagnostics["classification_query"].square().mean()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(refined).all())
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertTrue(torch.isfinite(module.evidence.logit.weight.grad).all())
        self.assertTrue(torch.isfinite(module.class_descriptor_head[-1].weight.grad).all())


class DFINETransformerIntegrationTest(unittest.TestCase):
    def test_cmt_keeps_official_inference_output_contract(self):
        decoder_module = importlib.import_module("src.zoo.dfine.dfine_decoder")
        decoder = decoder_module.DFINETransformer(
            num_classes=3,
            hidden_dim=16,
            num_queries=5,
            feat_channels=[16],
            feat_strides=[8],
            num_levels=1,
            num_points=2,
            nhead=4,
            num_layers=2,
            dim_feedforward=32,
            num_denoising=0,
            eval_spatial_size=[32, 32],
            reg_max=8,
        )
        branch = cmt.CMTMomentRefiner(
            hidden_dim=16,
            evidence_channels=4,
            num_moment_levels=2,
            reg_max=8,
            solver_steps=2,
        )
        images = torch.rand(2, 3, 32, 32)
        features = [torch.rand(2, 16, 4, 4)]

        decoder.train()
        train_output = decoder(features, cmt=branch, cmt_state=branch.prepare(images))
        self.assertEqual(train_output["pred_boxes"].shape, (2, 5, 4))
        self.assertIn("cmt", train_output)
        self.assertNotIn("classification_query", train_output["cmt"])

        decoder.eval()
        eval_output = decoder(features, cmt=branch, cmt_state=branch.prepare(images))
        self.assertEqual(set(eval_output), {"pred_logits", "pred_boxes"})

        official_output = decoder(features)
        baseline = cmt.CMTMomentRefiner(hidden_dim=16, reg_max=8, mode="baseline")
        baseline_output = decoder(
            features, cmt=baseline, cmt_state=baseline.prepare(images)
        )
        for key in official_output:
            self.assertTrue(torch.equal(official_output[key], baseline_output[key]))

        early_only = cmt.CMTMomentRefiner(
            hidden_dim=16,
            evidence_channels=4,
            num_moment_levels=2,
            reg_max=8,
            solver_steps=2,
            refine_layers=[0],
        )
        decoder.train()
        early_output = decoder(
            features, cmt=early_only, cmt_state=early_only.prepare(images)
        )
        self.assertIn("cmt", early_output)


if __name__ == "__main__":
    unittest.main()
