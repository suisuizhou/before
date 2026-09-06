import unittest

import torch


class TargetReliabilityTests(unittest.TestCase):
    def test_teacher_views_are_four_reproducible_and_clean_first(self):
        from sde_evmt_r18.target_views import make_teacher_views

        x = torch.linspace(0.0, 1.0, 64).repeat(3, 1)
        cfg = {
            "teacher_views": {
                "style_strength": 0.05,
                "warp_max": 0.5,
                "noise_snr_min": 35.0,
                "noise_snr_max": 45.0,
                "gain_min": 0.97,
                "gain_max": 1.03,
            }
        }
        first = make_teacher_views(x, cfg, torch.Generator().manual_seed(17))
        second = make_teacher_views(x, cfg, torch.Generator().manual_seed(17))

        self.assertEqual(len(first), 4)
        self.assertTrue(torch.equal(first[0], x))
        for left, right in zip(first, second):
            self.assertEqual(tuple(left.shape), tuple(x.shape))
            self.assertTrue(torch.equal(left, right))

    def test_margin_evidence_masks_salient_contiguous_band_and_reduces_margin(self):
        from sde_evmt_r18.evidence import verify_margin_evidence

        spectrum = torch.ones(2, 1, 64)
        spectrum[:, :, 20:28] = 10.0

        def forward_fn(value):
            score = value[:, :, 20:28].sum(dim=(1, 2))
            return torch.stack((score, -score), dim=1)

        logits = forward_fn(spectrum)
        result = verify_margin_evidence(
            forward_fn,
            spectrum,
            logits,
            torch.zeros(2, dtype=torch.long),
            {
                "evidence": {
                    "smooth_kernel": 3,
                    "num_bands": 1,
                    "mask_ratio_min": 0.125,
                    "mask_ratio_max": 0.125,
                }
            },
        )

        self.assertEqual(tuple(result.mask.shape), (2, 64))
        self.assertEqual(tuple(result.destroyed.shape), tuple(spectrum.shape))
        self.assertTrue((result.mask[:, 20:28].sum(dim=1) >= 6).all())
        self.assertTrue((result.margin_drop > 0).all())
        self.assertTrue(torch.isfinite(result.margin_drop).all())

    def test_router_uses_classwise_median_and_excludes_small_classes_from_memory(self):
        from sde_evmt_r18.reliability import EvidenceReliabilityRouter

        class_zero = torch.tensor([[0.95, 0.05]]).repeat(4, 1)
        class_one = torch.tensor([[0.05, 0.95]]).repeat(4, 1)
        mean_q = torch.cat((class_zero, class_one), dim=0)
        q_views = mean_q.unsqueeze(0).repeat(4, 1, 1)
        margin_drop = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4])
        router = EvidenceReliabilityRouter(
            confidence_min=0.0,
            gamma=5.0,
            min_view_agreement=0.75,
            min_class_samples=4,
        )

        route = router.route(q_views, margin_drop)

        self.assertEqual(int(route.certain[:4].sum()), 2)
        self.assertEqual(int(route.certain[4:].sum()), 2)
        self.assertTrue(torch.equal(route.memory_eligible, route.certain))
        self.assertTrue(torch.isfinite(route.reliability).all())

        small = router.route(q_views[:, :3], margin_drop[:3])
        self.assertFalse(bool(small.memory_eligible.any()))

    def test_router_requires_three_of_four_view_predictions_to_agree(self):
        from sde_evmt_r18.reliability import EvidenceReliabilityRouter

        q_views = torch.tensor(
            [
                [[0.9, 0.1]],
                [[0.8, 0.2]],
                [[0.2, 0.8]],
                [[0.1, 0.9]],
            ]
        )
        route = EvidenceReliabilityRouter(
            confidence_min=0.0,
            min_view_agreement=0.75,
            min_class_samples=1,
        ).route(q_views, torch.tensor([1.0]))

        self.assertEqual(float(route.view_vote_fraction[0]), 0.5)
        self.assertFalse(bool(route.certain[0]))
        self.assertFalse(bool(route.memory_eligible[0]))


if __name__ == "__main__":
    unittest.main()
