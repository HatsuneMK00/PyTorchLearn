import unittest
import numpy as np
from ..occlusion_bound import calculate_entire_bounds

class MarabouOcclusionTest(unittest.TestCase):
    # test the marabou_occlusion.py script
    # occlusion size = (1, 1), image size = (3, 3), epsilon = 0.5, occlusion point = (1, 1)
    def test_occlusion_1_1_3_3_0_5_1_1(self):
        # create a 3x3*3 image with value 4
        image = np.ones((3, 3, 3)) * 4
        image_origin = np.copy(image)
        upper_bounds, lower_bounds, changed = calculate_entire_bounds(image, (0.5, 0.5), (1, 1), 0, (0, 0), (3, 3), 0.5)

        # --- assert the bounds ---
        # f(0, 0) >= 3 / 4 * f(0, 0), f(0, 0) <= f(0, 0)
        # f(1, 0) >= 1 / 2 * f(1, 0), f(1, 0) <= f(1, 0)
        # f(2, 0) >= 3 / 4 * f(2, 0), f(2, 0) <= f(2, 0)
        # f(0, 1) >= 1 / 2 * f(0, 1), f(0, 1) <= f(0, 1)
        # f(1, 1) >= 0, f(1, 1) <= 3 / 4 * f(1, 1)
        # f(2, 1) >= 1 / 2 * f(2, 1), f(2, 1) <= f(2, 1)
        # f(0, 2) >= 3 / 4 * f(0, 2), f(0, 2) <= f(0, 2)
        # f(1, 2) >= 1 / 2 * f(1, 2), f(1, 2) <= f(1, 2)
        # f(2, 2) >= 3 / 4 * f(2, 2), f(2, 2) <= f(2, 2)
        # --- assert the bounds ---
        self.assertTrue((upper_bounds[0, 0] == image_origin[0, 0]).all())
        self.assertTrue((lower_bounds[0, 0] == 3 / 4 * image_origin[0, 0]).all())
        self.assertTrue((upper_bounds[0, 1] == image_origin[0, 1]).all())
        self.assertTrue((lower_bounds[0, 1] == 1 / 2 * image_origin[0, 1]).all())
        self.assertTrue((upper_bounds[0, 2] == image_origin[0, 2]).all())
        self.assertTrue((lower_bounds[0, 2] == 3 / 4 * image_origin[0, 2]).all())
        self.assertTrue((upper_bounds[1, 0] == image_origin[1, 0]).all())
        self.assertTrue((lower_bounds[1, 0] == 1 / 2 * image_origin[1, 0]).all())
        self.assertTrue((upper_bounds[1, 1] == 3 / 4 * image_origin[1, 1]).all())
        self.assertTrue((lower_bounds[1, 1] == 0).all())
        self.assertTrue((upper_bounds[1, 2] == image_origin[1, 2]).all())
        self.assertTrue((lower_bounds[1, 2] == 1 / 2 * image_origin[1, 2]).all())
        self.assertTrue((upper_bounds[2, 0] == image_origin[2, 0]).all())
        self.assertTrue((lower_bounds[2, 0] == 3 / 4 * image_origin[2, 0]).all())
        self.assertTrue((upper_bounds[2, 1] == image_origin[2, 1]).all())
        self.assertTrue((lower_bounds[2, 1] == 1 / 2 * image_origin[2, 1]).all())
        self.assertTrue((upper_bounds[2, 2] == image_origin[2, 2]).all())
        self.assertTrue((lower_bounds[2, 2] == 3 / 4 * image_origin[2, 2]).all())




if __name__ == '__main__':
    unittest.main()
