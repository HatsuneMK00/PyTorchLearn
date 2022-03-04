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
        upper_bounds, lower_bounds = calculate_entire_bounds(image, (0.5, 0.5), (1, 1), 0, (0, 0), (2, 2), 0.5)

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


    # test the marabou_occlusion.py script
    # occlusion size = (2, 2), image size = (4, 4), epsilon = 0.5, occlusion point = (1, 1)
    def test_occlusion_2_2_4_4_0_5_1_1(self):
        # create a 4x4*3 image with value 4
        image = np.ones((4, 4, 3)) * 4
        image_origin = np.copy(image)
        upper_bounds, lower_bounds = calculate_entire_bounds(image, (0.5, 0.5), (2, 2), 0, (0, 0), (3, 3), 0.5)

        # --- assert the bounds ---
        # f(0, 0) >= 3 / 4 * f(0, 0), f(0, 0) <= f(0, 0)
        # f(1, 0) >= 1 / 2 * f(1, 0), f(1, 0) <= f(1, 0)
        # f(2, 0) >= 1 / 2 * f(2, 0), f(2, 0) <= f(2, 0)
        # f(3, 0) >= 3 / 4 * f(3, 0), f(3, 0) <= f(3, 0)
        # f(0, 1) >= 1 / 2 * f(0, 1), f(0, 1) <= f(0, 1)
        # f(1, 1) >= 0, f(1, 1) <= 3 / 4 * f(1, 1)
        # f(2, 1) >= 0, f(2, 1) <= 3 / 4 * f(2, 1)
        # f(3, 1) >= 1 / 2 * f(3, 1), f(3, 1) <= f(3, 1)
        # f(*, 2) is same as f(*, 1)
        # f(*, 3) is same as f(*, 0)
        # --- assert the bounds ---
        self.assertTrue((upper_bounds[0, 0] == image_origin[0, 0]).all())
        self.assertTrue((lower_bounds[0, 0] == 3 / 4 * image_origin[0, 0]).all())
        self.assertTrue((upper_bounds[0, 1] == image_origin[0, 1]).all())
        self.assertTrue((lower_bounds[0, 1] == 1 / 2 * image_origin[0, 1]).all())
        self.assertTrue((upper_bounds[0, 2] == image_origin[0, 2]).all())
        self.assertTrue((lower_bounds[0, 2] == 1 / 2 * image_origin[0, 2]).all())
        self.assertTrue((upper_bounds[0, 3] == image_origin[0, 3]).all())
        self.assertTrue((lower_bounds[0, 3] == 3 / 4 * image_origin[0, 3]).all())
        self.assertTrue((upper_bounds[1, 0] == image_origin[1, 0]).all())
        self.assertTrue((lower_bounds[1, 0] == 1 / 2 * image_origin[1, 0]).all())
        self.assertTrue((upper_bounds[1, 1] == 3 / 4 * image_origin[1, 1]).all())
        self.assertTrue((lower_bounds[1, 1] == 0).all())
        self.assertTrue((upper_bounds[1, 2] == 3 / 4 * image_origin[1, 2]).all())
        self.assertTrue((lower_bounds[1, 2] == 0).all())
        self.assertTrue((upper_bounds[1, 3] == image_origin[1, 3]).all())
        self.assertTrue((lower_bounds[1, 3] == 1 / 2 * image_origin[1, 3]).all())

        self.assertTrue((upper_bounds[3, 0] == image_origin[3, 0]).all())
        self.assertTrue((lower_bounds[3, 0] == 3 / 4 * image_origin[3, 0]).all())
        self.assertTrue((upper_bounds[3, 1] == image_origin[3, 1]).all())
        self.assertTrue((lower_bounds[3, 1] == 1 / 2 * image_origin[3, 1]).all())
        self.assertTrue((upper_bounds[3, 2] == image_origin[3, 2]).all())
        self.assertTrue((lower_bounds[3, 2] == 1 / 2 * image_origin[3, 2]).all())
        self.assertTrue((upper_bounds[3, 3] == image_origin[3, 3]).all())
        self.assertTrue((lower_bounds[3, 3] == 3 / 4 * image_origin[3, 3]).all())
        self.assertTrue((upper_bounds[2, 0] == image_origin[2, 0]).all())
        self.assertTrue((lower_bounds[2, 0] == 1 / 2 * image_origin[2, 0]).all())
        self.assertTrue((upper_bounds[2, 1] == 3 / 4 * image_origin[2, 1]).all())
        self.assertTrue((lower_bounds[2, 1] == 0).all())
        self.assertTrue((upper_bounds[2, 2] == 3 / 4 * image_origin[2, 2]).all())
        self.assertTrue((lower_bounds[2, 2] == 0).all())
        self.assertTrue((upper_bounds[2, 3] == image_origin[2, 3]).all())
        self.assertTrue((lower_bounds[2, 3] == 1 / 2 * image_origin[2, 3]).all())

    # test the marabou_occlusion.py script
    # occlusion size = (2, 1), image size = (4, 4), epsilon = 0.5, occlusion point = (1, 1)
    def test_occlusion_2_1_4_4_0_5_1_1(self):
        # create a 4x4*3 image with value 4
        image = np.ones((4, 4, 3)) * 4
        image_origin = np.copy(image)
        upper_bounds, lower_bounds = calculate_entire_bounds(image, (0.5, 0.5), (1, 2), 0, (0, 0), (2, 3), 0.5)

        # --- assert the bounds ---
        # f(0, 0) >= 3 / 4 * f(0, 0), f(0, 0) <= f(0, 0)
        # f(1, 0) >= 1 / 2 * f(1, 0), f(1, 0) <= f(1, 0)
        # f(2, 0) >= 1 / 2 * f(2, 0), f(2, 0) <= f(2, 0)
        # f(3, 0) >= 3 / 4 * f(3, 0), f(3, 0) <= f(3, 0)
        # f(0, 1) >= 1 / 2 * f(0, 1), f(0, 1) <= f(0, 1)
        # f(1, 1) >= 0, f(1, 1) <= 3 / 4 * f(1, 1)
        # f(2, 1) >= 0, f(2, 1) <= 3 / 4 * f(2, 1)
        # f(3, 1) >= 1 / 2 * f(3, 1), f(3, 1) <= f(3, 1)
        # f(0, 2) >= 3 / 4 * f(0, 2), f(0, 2) <= f(0, 2)
        # f(1, 2) >= 1 / 2 * f(1, 2), f(1, 2) <= f(1, 2)
        # f(2, 2) >= 1 / 2 * f(2, 2), f(2, 2) <= f(2, 2)
        # f(3, 2) >= 3 / 4 * f(3, 2), f(3, 2) <= f(3, 2)
        # f(0, 3) >= f(0, 3), f(0, 3) <= f(0, 3)
        # f(1, 3) >= f(1, 3), f(1, 3) <= f(1, 3)
        # f(2, 3) >= f(2, 3), f(2, 3) <= f(2, 3)
        # f(3, 3) >= f(3, 3), f(3, 3) <= f(3, 3)
        # --- assert the bounds ---
        self.assertTrue((upper_bounds[0, 0] == image_origin[0, 0]).all())
        self.assertTrue((lower_bounds[0, 0] == 3 / 4 * image_origin[0, 0]).all())
        self.assertTrue((upper_bounds[0, 1] == image_origin[0, 1]).all())
        self.assertTrue((lower_bounds[0, 1] == 1 / 2 * image_origin[0, 1]).all())
        self.assertTrue((upper_bounds[0, 2] == image_origin[0, 2]).all())
        self.assertTrue((lower_bounds[0, 2] == 1 / 2 * image_origin[0, 2]).all())
        self.assertTrue((upper_bounds[0, 3] == image_origin[0, 3]).all())
        self.assertTrue((lower_bounds[0, 3] == 3 / 4 * image_origin[0, 3]).all())
        self.assertTrue((upper_bounds[1, 0] == image_origin[1, 0]).all())
        self.assertTrue((lower_bounds[1, 0] == 1 / 2 * image_origin[1, 0]).all())
        self.assertTrue((upper_bounds[1, 1] == 3 / 4 * image_origin[1, 1]).all())
        self.assertTrue((lower_bounds[1, 1] == 0).all())
        self.assertTrue((upper_bounds[1, 2] == 3 / 4 * image_origin[1, 2]).all())
        self.assertTrue((lower_bounds[1, 2] == 0).all())
        self.assertTrue((upper_bounds[1, 3] == image_origin[1, 3]).all())
        self.assertTrue((lower_bounds[1, 3] == 1 / 2 * image_origin[1, 3]).all())

        self.assertTrue((upper_bounds[3, 0] == image_origin[3, 0]).all())
        self.assertTrue((lower_bounds[3, 0] == image_origin[3, 0]).all())
        self.assertTrue((upper_bounds[3, 1] == image_origin[3, 1]).all())
        self.assertTrue((lower_bounds[3, 1] == image_origin[3, 1]).all())
        self.assertTrue((upper_bounds[3, 2] == image_origin[3, 2]).all())
        self.assertTrue((lower_bounds[3, 2] == image_origin[3, 2]).all())
        self.assertTrue((upper_bounds[3, 3] == image_origin[3, 3]).all())
        self.assertTrue((lower_bounds[3, 3] == image_origin[3, 3]).all())
        self.assertTrue((upper_bounds[2, 0] == image_origin[2, 0]).all())
        self.assertTrue((lower_bounds[2, 0] == 3 / 4 * image_origin[2, 0]).all())
        self.assertTrue((upper_bounds[2, 1] == image_origin[2, 1]).all())
        self.assertTrue((lower_bounds[2, 1] == 1 / 2 * image_origin[2, 1]).all())
        self.assertTrue((upper_bounds[2, 2] == image_origin[2, 2]).all())
        self.assertTrue((lower_bounds[2, 2] == 1 / 2 * image_origin[2, 2]).all())
        self.assertTrue((upper_bounds[2, 3] == image_origin[2, 3]).all())
        self.assertTrue((lower_bounds[2, 3] == 3 / 4 * image_origin[2, 3]).all())


if __name__ == '__main__':
    unittest.main()
