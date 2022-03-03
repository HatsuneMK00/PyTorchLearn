# -*- coding: utf-8 -*-
# created by makise, 2022/3/3

import numpy as np
from interpolation.occlusion import occlusion

def calculate_entire_bounds(image, left_upper_occ, occlusion_size, occlusion_color, left_upper_affected,
                            affected_size, epsilon):
    """
    Calculate the all the bounds of network
    :param image: np array with size (height, width, 3)
    :param left_upper_occ: left upper occlusion point
    :param occlusion_size: 2-tuple indicating the occlusion size, must be integer
    :param occlusion_color: int indicating the occlusion color, must be in range [0, 255]
    :param left_upper_affected: left upper point of affected area
    :param affected_size: 2-tuple indicating the affected area size, must be integer
    :param epsilon: float indicating the epsilon, must be positive, can be non-integer
    :return: (upper_bounds(32*32*3), lower_bounds(32*32*3), changed(32*32))
    """
    # --------------------------------------------------
    # for every optimal-possible occlusion point on the edge of occlusion area
    # calculate the pixel value in the affected area
    # --------------------------------------------------
    # first create variables for recording the upper and lower bounds for each pixel
    # record the upper bound for pixels that are affected by the occlusion area to find the minimum
    # and the lower bound for pixels that are affected by the occlusion area to find the maximum
    # the upper bound and lower bound has same size as the image
    upper_bounds = np.zeros(image.shape)
    lower_bounds = np.zeros(image.shape)
    # record whether the upper bound and lower bound is changed for each pixel
    # has same size as the image but without last dimension
    changed = np.zeros(image.shape[:-1])

    # --------------------------------------------------
    # iterate through optimal-possible occlusion point on the edge of occlusion area
    # --------------------------------------------------
    # iterate through the upper parallel edge of possible occlusion area
    current_left_upper_occ = left_upper_occ
    while current_left_upper_occ[0] < left_upper_occ[0] + 2 * epsilon:
        # update the upper_bounds, lower_bounds and changed according to the current occlusion point,
        # occlusion size and affected area
        update_bounds(image, current_left_upper_occ, occlusion_size, occlusion_color, left_upper_affected,
                      affected_size, upper_bounds, lower_bounds, changed)
        # next occlusion point
        current_left_upper_occ = (np.floor(current_left_upper_occ[0] + 1), current_left_upper_occ[1])
    # complement the last right upper occlusion point
    current_left_upper_occ = (left_upper_occ[0] + 2 * epsilon, current_left_upper_occ[1])
    update_bounds(image, current_left_upper_occ, occlusion_size, occlusion_color, left_upper_affected,
                  affected_size, upper_bounds, lower_bounds, changed)

    # iterate through the lower parallel edge of possible occlusion area
    current_left_upper_occ = (left_upper_occ[0], left_upper_occ[1] + 2 * epsilon)
    while current_left_upper_occ[0] < left_upper_occ[0] + 2 * epsilon:
        # update the upper_bounds, lower_bounds and changed according to the current occlusion point,
        # occlusion size and affected area
        update_bounds(image, current_left_upper_occ, occlusion_size, occlusion_color, left_upper_affected,
                      affected_size, upper_bounds, lower_bounds, changed)
        # next occlusion point
        current_left_upper_occ = (np.floor(current_left_upper_occ[0] + 1), current_left_upper_occ[1])
    # complement the last right lower occlusion point
    current_left_upper_occ = (left_upper_occ[0] + 2 * epsilon, current_left_upper_occ[1])
    update_bounds(image, current_left_upper_occ, occlusion_size, occlusion_color, left_upper_affected,
                  affected_size, upper_bounds, lower_bounds, changed)

    # iterate through the left vertical edge of possible occlusion area
    current_left_upper_occ = left_upper_occ
    while current_left_upper_occ[1] < left_upper_occ[1] + 2 * epsilon:
        # update the upper_bounds, lower_bounds and changed according to the current occlusion point,
        # occlusion size and affected area
        update_bounds(image, current_left_upper_occ, occlusion_size, occlusion_color, left_upper_affected,
                      affected_size, upper_bounds, lower_bounds, changed)
        # next occlusion point
        current_left_upper_occ = (current_left_upper_occ[0], np.floor(current_left_upper_occ[1] + 1))
    # complement the last left upper occlusion point
    current_left_upper_occ = (current_left_upper_occ[0], left_upper_occ[1] + 2 * epsilon)
    update_bounds(image, current_left_upper_occ, occlusion_size, occlusion_color, left_upper_affected,
                  affected_size, upper_bounds, lower_bounds, changed)

    # iterate through the right vertical edge of possible occlusion area
    current_left_upper_occ = (left_upper_occ[0] + 2 * epsilon, left_upper_occ[1])
    while current_left_upper_occ[1] < left_upper_occ[1] + 2 * epsilon:
        # update the upper_bounds, lower_bounds and changed according to the current occlusion point,
        # occlusion size and affected area
        update_bounds(image, current_left_upper_occ, occlusion_size, occlusion_color, left_upper_affected,
                      affected_size, upper_bounds, lower_bounds, changed)
        # next occlusion point
        current_left_upper_occ = (current_left_upper_occ[0], np.floor(current_left_upper_occ[1] + 1))
    # complement the last left lower occlusion point
    current_left_upper_occ = (current_left_upper_occ[0], left_upper_occ[1] + 2 * epsilon)
    update_bounds(image, current_left_upper_occ, occlusion_size, occlusion_color, left_upper_affected,
                  affected_size, upper_bounds, lower_bounds, changed)

    # iterate through points covered by the occlusion area and set the lower_bounds to occlusion color
    for i in range(int(np.ceil(left_upper_occ[0])), occlusion_size[0] + 1):
        for j in range(int(np.ceil(left_upper_occ[1])), occlusion_size[1] + 1):
            lower_bounds[i][j] = occlusion_color

    return upper_bounds, lower_bounds, changed


def update_bounds(image, left_upper_occ, occlusion_size, occlusion_color, left_upper_affected, affected_size,
                  upper_bounds, lower_bounds, changed):
    """
    update the upper_bounds, lower_bounds and changed according to the current occlusion point,
    occlusion size and affected area
    :param image: image in np array, 32*32*3
    :param left_upper_occ: the left upper point of occlusion area, can be non-integer
    :param occlusion_size: the (height, width) of occlusion area
    :param occlusion_color: the color of occlusion area
    :param left_upper_affected: the left upper point of affected area, must be integer
    :param affected_size: the (height, width) of affected area
    :param upper_bounds: the upper_bounds np array needs to be updated
    :param lower_bounds: the lower_bounds np array needs to be updated
    :param changed: the changed np array needs to be updated, size is 32*32
    :return:
    """
    # --------------------------------------------------
    # calculate the image under this occlusion settings
    # --------------------------------------------------
    img_np = occlusion(image, left_upper_occ, occlusion_size, occlusion_color)
    h, w, c = img_np.shape
    # iterate through the affected area to update the upper_bounds and lower_bounds
    for i in range(left_upper_affected[0], left_upper_affected[0] + affected_size[0] + 1):
        for j in range(left_upper_affected[1], left_upper_affected[1] + affected_size[1] + 1):
            # get the color of the pixel
            pixel = img_np[i, j]
            if not changed[i, j]:
                changed[i, j] = True
                # update the upper_bounds and lower_bounds
                upper_bounds[i, j] = pixel
                lower_bounds[i, j] = pixel
            else:
                # update the upper_bounds and lower_bounds for each channel
                for k in range(c):
                    if pixel[k] > upper_bounds[i, j][k]:
                        upper_bounds[i, j][k] = pixel[k]
                    if pixel[k] < lower_bounds[i, j][k]:
                        lower_bounds[i, j][k] = pixel[k]
