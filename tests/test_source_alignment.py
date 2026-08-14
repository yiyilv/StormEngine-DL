import unittest

import numpy as np

from stormengine_dl.source_alignment import bilinear_sample_grid, paired_source_statistics


class SourceAlignmentTests(unittest.TestCase):
    def test_bilinear_sampling_preserves_exact_and_centre_values(self) -> None:
        fields = np.asarray([[[[0.0, 2.0], [4.0, 6.0]]]], dtype=np.float32)
        sampled = bilinear_sample_grid(
            fields, np.asarray([0.0, 1.0]), np.asarray([10.0, 11.0]),
            np.asarray([[0.0, 10.0], [0.5, 10.5], [1.0, 11.0]]),
        )
        self.assertEqual(sampled.shape, (1, 3, 1))
        np.testing.assert_allclose(sampled[0, :, 0], [0.0, 3.0, 6.0])

    def test_paired_statistics_respect_mask_and_bias_sign(self) -> None:
        reference = np.asarray([[[1.0, 1.0], [2.0, 5.0]], [[3.0, 7.0], [4.0, 9.0]]])
        source = reference + np.asarray([1.0, -2.0])
        mask = np.ones_like(source, dtype=bool); mask[0, 1, 0] = False
        result = paired_source_statistics(source, reference, mask, ("a", "b"))
        self.assertEqual(result["a"]["count"], 3)
        self.assertAlmostEqual(result["a"]["bias"], 1.0)
        self.assertAlmostEqual(result["a"]["rmse"], 1.0)
        self.assertAlmostEqual(result["a"]["correlation"], 1.0)
        self.assertAlmostEqual(result["b"]["bias"], -2.0)
        self.assertAlmostEqual(result["b"]["rmse"], 2.0)

    def test_rejects_points_outside_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            bilinear_sample_grid(
                np.zeros((1, 1, 2, 2)), np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]),
                np.asarray([[2.0, 0.5]]),
            )


if __name__ == "__main__": unittest.main()
