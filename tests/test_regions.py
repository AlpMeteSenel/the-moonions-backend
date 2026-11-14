import unittest

import numpy as np

from moonions_backend.regions import _zones


class LandingEllipseTests(unittest.TestCase):
    def zones(self, grid, radius=180):
        grid = np.asarray(grid, dtype=float)
        axis = np.arange(grid.size, dtype=float)
        return _zones(grid.ravel(), grid.shape[0], 250, radius, axis, axis, axis, axis)

    def test_isolated_cell_cannot_contain_landing_ellipse(self):
        grid = np.zeros((5, 5))
        grid[2, 2] = 0.92
        self.assertEqual(self.zones(grid), [])

    def test_half_size_ellipse_is_not_enough(self):
        grid = np.zeros((5, 5))
        grid[2, 1:4] = 0.95
        self.assertEqual(self.zones(grid), [])

    def test_region_boundary_limits_clearance(self):
        zone = self.zones(np.ones((3, 3)))[0]
        self.assertEqual(zone["margin_m"], 250)
        self.assertEqual(zone["east_m"], 4)  # central cell, not an unbounded edge
        self.assertTrue(zone["fits_ellipse"])

    def test_large_patch_fits_and_small_patch_is_excluded(self):
        grid = np.zeros((9, 9))
        grid[1:4, 1:4] = 0.9
        grid[7, 7] = 0.99
        zones = self.zones(grid)
        self.assertEqual(len(zones), 1)
        self.assertEqual(zones[0]["id"], "Z1")
        self.assertGreaterEqual(zones[0]["margin_m"], 180)
        self.assertAlmostEqual(zones[0]["mean_success"], 0.9)

    def test_no_safe_ground_has_no_zones(self):
        self.assertEqual(self.zones(np.zeros((5, 5))), [])

    def test_ellipse_larger_than_region_is_rejected(self):
        self.assertEqual(self.zones(np.ones((3, 3)), radius=300), [])


if __name__ == "__main__":
    unittest.main()
