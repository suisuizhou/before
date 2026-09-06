import unittest

from main_tta_evmt import method_overrides


class MethodVariantTests(unittest.TestCase):
    def test_bn_stat_has_no_optimizer_updates(self):
        cfg = method_overrides("bn_stat")

        self.assertTrue(cfg.use_bn_stats)
        self.assertFalse(cfg.train_bn_affine)
        self.assertFalse(cfg.train_adapter)
        self.assertFalse(cfg.train_warp)
        self.assertFalse(cfg.use_mt)
        self.assertFalse(cfg.use_evidence)

    def test_bn_affine_adds_only_bn_gradient_adaptation(self):
        cfg = method_overrides("bn_affine")

        self.assertTrue(cfg.train_bn_affine)
        self.assertFalse(cfg.train_adapter)
        self.assertFalse(cfg.train_warp)
        self.assertFalse(cfg.use_mt)

    def test_bn_mt_adapter_enables_lightweight_parameters_and_teacher(self):
        cfg = method_overrides("bn_mt_adapter")

        self.assertTrue(cfg.train_bn_affine)
        self.assertTrue(cfg.train_adapter)
        self.assertTrue(cfg.train_warp)
        self.assertTrue(cfg.use_mt)
        self.assertFalse(cfg.use_evidence)
        self.assertFalse(cfg.use_pcl)
        self.assertFalse(cfg.use_ncl)

    def test_full_enables_state_driven_components(self):
        cfg = method_overrides("bnfirst_full")

        self.assertTrue(cfg.train_bn_affine)
        self.assertTrue(cfg.use_evidence)
        self.assertTrue(cfg.use_pcl)
        self.assertTrue(cfg.use_ncl)

    def test_unknown_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown BN-first method"):
            method_overrides("mystery")


if __name__ == "__main__":
    unittest.main()
