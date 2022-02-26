# -*- coding: utf-8 -*-
# created by makise, 2022/2/26

"""
    interpolation/testing.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    testing interpolation
"""

from occlusion import *
from PIL import Image

import matplotlib.pyplot as plt

# load one sample image from gtsrb dataset using PIL
def load_sample_image():
    path = '../data/GTSRB/trainingset/00000/00000_00000.ppm'
    img = Image.open(path)
    return img


def regular_occlusion_test():
    img = load_sample_image()
    occlusion_size = (10, 10)
    occlusion_image = regular_occlusion(img, (3, 3), occlusion_size, 0)

    # occlusion using PIL paste() as control image
    # generate a black occlusion mask
    mask = Image.new('L', occlusion_size, 0)
    # paste the occlusion mask to the image
    img.paste(mask, (3, 3))

    # compare the two images using plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(occlusion_image)
    plt.show()

    # assert the two images are the same
    assert np.array_equal(np.array(img), np.array(occlusion_image))


def non_integer_occlusion_test():
    img = load_sample_image()
    occlusion_size = (10, 10)
    occlusion_image = occlusion_with_interpolation(img, (3.5, 3.5), occlusion_size, 0)

    # occlusion using PIL paste() as control image
    # generate a black occlusion mask
    mask = Image.new('L', occlusion_size, 0)
    # paste the occlusion mask to the image
    img.paste(mask, (3, 3))

    # compare the two images using plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(occlusion_image)
    plt.show()


if __name__ == '__main__':
    # regular_occlusion_test()
    non_integer_occlusion_test()