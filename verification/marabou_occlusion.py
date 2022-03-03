# -*- coding: utf-8 -*-
# created by makise, 2022/3/1

# This script is used to verify occlusion type perturbation with Marabou.


import onnx
import onnxruntime
from maraboupy import Marabou, MarabouNetwork
from PIL import Image
import numpy as np

from marabou_utils import load_network, load_sample_image
from occlusion_bound import calculate_entire_bounds


# verify with marabou
def verify_with_marabou(network: MarabouNetwork, image: np.array, label: int, box, occlusion_size=(1, 1),
                        occlusion_color=0, epsilon=0.5):
    """
    Verify occlusion on image with marabou
    :param network: MarabouNetwork
    :param image: a 1*3*32*32 image in np array
    :param label: int indicates the correct label
    :param box: a 2-tuple indicates the left upper point of occlusion area
    :param occlusion_size: a 2-tuple integer indicates the height and width of occlusion area. Default to (1, 1)
    :param occlusion_color: int indicates the color of occlusion area. Default to 0
    :param epsilon: float indicates how much the occlusion area would move. Default to 0.5
    """
    inputs = network.inputVars[0][0]  # the first dimension is batch size which is 1
    n_inputs = inputs.flatten().shape[0]
    inputs_flattened = inputs.flatten()
    print("input_shape:", inputs.shape)
    outputs = network.outputVars[0]
    n_outputs = outputs.flatten().shape[0]
    outputs_flattened = outputs.flatten()
    print("output_shape:", outputs.shape)
    image = image[0]
    image_flattened = image.flatten()

    # unpack inputs size (channel, height, width)
    c, h, w = inputs.shape
    # unpack the occlusion size
    h_o, w_o = occlusion_size
    # unpack the left upper point of occlusion area
    x, y = box
    # assert image has the same size with inputs
    assert image.shape == (c, h, w)

    # transpose the image to (height, width, channel)
    # todo the size of image and inputs are not same, same should be better
    image = np.transpose(image, (1, 2, 0))
    assert image.shape == (h, w, c)

    # set the all inputs to the image values
    for i in range(n_inputs):
        val = image_flattened[i]
        network.setLowerBound(inputs_flattened[i], val)
        network.setUpperBound(inputs_flattened[i], val)

    # set the possible occlusion area
    # since the occlusion area can move up to epsilon, the possible occlusion
    # area is a square with size (h_o + 2 * epsilon, w_o + 2 * epsilon) and the left upper point
    # is (x - epsilon, y - epsilon)
    # only pixels around the edge of the square contributes to the maximum and minimum of bounds
    # the edge of the square are (x - epsilon, y - epsilon) to (x + epsilon + w_o, y - epsilon)
    # and (x - epsilon, y - epsilon) to (x - epsilon, y + epsilon + h_o)
    # and (x - epsilon, y + epsilon + h_o) to (x + epsilon + w_o, y + epsilon + h_o)
    # and (x + epsilon + w_o, y - epsilon) to (x + epsilon + w_o, y + epsilon + h_o)
    left_upper_occ = (x - epsilon, y - epsilon)
    height_occ = (h_o - 1) + 2 * epsilon
    width_occ = (w_o - 1) + 2 * epsilon
    left_lower_occ = (left_upper_occ[0], left_upper_occ[1] + height_occ)
    right_upper_occ = (left_upper_occ[0] + width_occ, left_upper_occ[1])
    right_lower_occ = (left_upper_occ[0] + width_occ, left_upper_occ[1] + height_occ)

    # set the affected area of occlusion as the floor of left_upper_occ
    # the affected area is larger than occlusion area by at most 1
    left_upper_affected = (np.floor(left_upper_occ[0]), np.floor(left_upper_occ[1]))
    left_lower_affected = (np.floor(left_lower_occ[0]), np.ceil(left_lower_occ[1]))
    right_upper_affected = (np.ceil(right_upper_occ[0]), np.floor(right_upper_occ[1]))
    right_lower_affected = (np.ceil(right_lower_occ[0]), np.ceil(right_lower_occ[1]))
    height_affected = left_lower_affected[1] - left_upper_affected[1]
    width_affected = right_upper_affected[0] - left_upper_affected[0]

    # todo assert the affected area is larger than occlusion area by at most 1

    upper_bounds, lower_bounds, changed = calculate_entire_bounds(image, left_upper_occ, occlusion_size,
                                                                  occlusion_color, left_upper_affected,
                                                                  (height_affected, width_affected), epsilon)

    # ------------------------------------------------------------------------------------------
    # set network input bounds according to lower_bounds, upper_bounds and changed
    # ------------------------------------------------------------------------------------------
    # iterate over changed
    for i in range(len(changed)):
        for j in range(len(changed[i])):
            if changed[i][j] == True:
                for channel in range(c):
                    network.setUpperBound(inputs[c][i][j], upper_bounds[i][j][channel])
                    network.setLowerBound(inputs[c][i][j], lower_bounds[i][j][channel])

    vals = network.solve(verbose=1)
    print(vals)


# test with some fixed upper and lower bounds
def verify_with_marabou_test(network: MarabouNetwork, image: np.array, label: int, box, occlusion_size=(1, 1),
                             epsilon=0.5):
    inputs = network.inputVars[0][0]  # the first dimension is batch size which is 1
    n_inputs = inputs.flatten().shape[0]
    inputs_flattened = inputs.flatten()
    print("input_shape:", inputs.shape)
    outputs = network.outputVars[0]
    n_outputs = outputs.flatten().shape[0]
    outputs_flattened = outputs.flatten()
    print("output_shape:", outputs.shape)
    image = image[0]
    image_flattened = image.flatten()

    # unpack inputs size (channel, height, width)
    c, h, w = inputs.shape
    # unpack the occlusion size
    h_o, w_o = occlusion_size
    # set x, y to the center of image
    x = int(w / 2)
    y = int(h / 2)
    # assert image has the same size with inputs
    assert image.shape == (c, h, w)

    # set all inputs as image value
    for i in range(n_inputs):
        val = image_flattened[i]
        network.setLowerBound(inputs_flattened[i], val)
        network.setUpperBound(inputs_flattened[i], val)

    # set the upper bound and lower bound for the pixels on the occlusion area for every channel
    for c in range(c):
        network.setLowerBound(inputs[c, y - 1, x - 1], 3.0 / 4 * image[c, y - 1, x - 1])
        network.setUpperBound(inputs[c, y - 1, x - 1], image[c, y - 1, x - 1])
        network.setLowerBound(inputs[c, y - 1, x], 2.0 / 4 * image[c, y - 1, x])
        network.setUpperBound(inputs[c, y - 1, x], image[c, y - 1, x])
        network.setLowerBound(inputs[c, y - 1, x + 1], 3.0 / 4 * image[c, y - 1, x + 1])
        network.setUpperBound(inputs[c, y - 1, x + 1], image[c, y - 1, x + 1])
        network.setLowerBound(inputs[c, y, x - 1], 2.0 / 4 * image[c, y, x - 1])
        network.setUpperBound(inputs[c, y, x - 1], image[c, y, x - 1])
        network.setLowerBound(inputs[c, y, x], 0)
        network.setUpperBound(inputs[c, y, x], 1.0 / 4 * image[c, y, x])
        network.setLowerBound(inputs[c, y, x + 1], 2.0 / 4 * image[c, y, x + 1])
        network.setUpperBound(inputs[c, y, x + 1], image[c, y, x + 1])
        network.setLowerBound(inputs[c, y + 1, x - 1], 3.0 / 4 * image[c, y + 1, x - 1])
        network.setUpperBound(inputs[c, y + 1, x - 1], image[c, y + 1, x - 1])
        network.setLowerBound(inputs[c, y + 1, x], 2.0 / 4 * image[c, y + 1, x])
        network.setUpperBound(inputs[c, y + 1, x], image[c, y + 1, x])
        network.setLowerBound(inputs[c, y + 1, x + 1], 3.0 / 4 * image[c, y + 1, x + 1])
        network.setUpperBound(inputs[c, y + 1, x + 1], image[c, y + 1, x + 1])

    # set the bounds for outputs
    for i in range(n_outputs):
        if i != label:
            network.setUpperBound(outputs_flattened[i], outputs_flattened[label])

    vals = network.solve(verbose=1)
    print("vals:", vals)


if __name__ == '__main__':
    # load sample image
    np_img = load_sample_image()

    # load network
    network = load_network('../model/fnn_model_gtsrb_small.onnx')
    label = 0
    verify_with_marabou_test(network, np_img, label, (0, 0), (1, 1))
