# -*- coding: utf-8 -*-
# created by makise, 2022/3/1

# This script is used to verify occlusion type perturbation with Marabou.


import onnx
import onnxruntime
from maraboupy import Marabou, MarabouNetwork
from PIL import Image
import numpy as np

from marabou_utils import load_network, load_sample_image


# verify with marabou
def verify_with_marabou(network:MarabouNetwork, image:np.array, label:int, box, occlusion_size=(1, 1), epsilon = 0.5):
    """
    Verify occlusion on image with marabou
    :param network: MarabouNetwork
    :param image: a 1*3*32*32 image in np array
    :param label: int indicates the correct label
    :param box: a 2-tuple indicates the left upper point of occlusion area
    :param occlusion_size: a 2-tuple indicates the height and width of occlusion area. Default to (1, 1)
    :param epsilon: float indicates how much the occlusion area would move. Default to 0.5
    """
    inputs = network.inputVars[0][0] # the first dimension is batch size which is 1
    print("input_shape:", inputs.shape)
    outputs = network.outputVars[0]
    print("output_shape:", outputs.shape)
    image = image[0]

    # unpack inputs size (channel, height, width)
    c, h, w = inputs.shape
    # unpack the occlusion size
    h_o, w_o = occlusion_size
    # unpack the left upper point of occlusion area
    x, y = box
    # assert image has the same size with inputs
    assert image.shape == (c, h, w)

    # set upper bound and lower bound for pixels that are affected by the occlusion area
    # which is a square with size (h_o, w_o)
    # since the occlusion area can move up to epsilon, the area that is affected by the occlusion
    # area is a square with size (h_o + 2 * epsilon, w_o + 2 * epsilon) and the left upper point
    # is (x - epsilon, y - epsilon)
    # and only pixels around the edge of the square need to set upper bound and lower bound
    # the edge of the square are (x - epsilon, y - epsilon) to (x + epsilon + w_o, y - epsilon)
    # and (x - epsilon, y - epsilon) to (x - epsilon, y + epsilon + h_o)
    # and (x - epsilon, y + epsilon + h_o) to (x + epsilon + w_o, y + epsilon + h_o)
    # and (x + epsilon + w_o, y - epsilon) to (x + epsilon + w_o, y + epsilon + h_o)

    # record the upper bound for pixels that are affected by the occlusion area to find the minimum
    # and the lower bound for pixels that are affected by the occlusion area to find the maximum
    # the upper bound and lower bound has same size as the inputs
    upper_bounds = np.zeros(inputs.shape)
    lower_bounds = np.zeros(inputs.shape)
    # record whether the upper bound and lower bound is changed for each pixel
    changed = np.zeros(inputs.shape)

    # iterate over the pixels on the left vertical line of the square
    i = x - epsilon
    j = y - epsilon
    while j < y + epsilon + h_o:
        # set the upper bound and lower bound for the pixels around the left vertical line for every channel
        for c in range(c):
            # set the upper bound, if smaller than the current upper bound, update the upper bound
            u_b = image[c, j, i]
            if u_b < upper_bounds[c, j, i]:
                upper_bounds[c, j, i] = u_b
            # set the lower bound
            l_b = image[c, j, i]
            if l_b > lower_bounds[c, j, i]:
                lower_bounds[c, j, i] = l_b
        j += 1
    # iterate over the pixels on the right vertical line of the square
    i = x + epsilon + w_o
    j = y - epsilon
    while j < y + epsilon + h_o:
        # set the upper bound and lower bound for the pixels on the right vertical line for every channel
        for c in range(c):
            # set the upper bound
            network.setUpperBound(inputs[c, i, j], 1)
            # set the lower bound
            network.setLowerBound(inputs[c, i, j], 0)
        j += 1
    # iterate over the pixels on the top horizontal line of the square
    i = x - epsilon
    j = y - epsilon
    while i < x + epsilon + w_o:
        # set the upper bound and lower bound for the pixels on the top horizontal line for every channel
        for c in range(c):
            # set the upper bound
            network.setUpperBound(inputs[c, i, j], 1)
            # set the lower bound
            network.setLowerBound(inputs[c, i, j], 0)
        i += 1
    # iterate over the pixels on the bottom horizontal line of the square
    i = x - epsilon
    j = y + epsilon + h_o
    while i < x + epsilon + w_o:
        # set the upper bound and lower bound for the pixels on the bottom horizontal line for every channel
        for c in range(c):
            # set the upper bound
            network.setUpperBound(inputs[c, i, j], 1)
            # set the lower bound
            network.setLowerBound(inputs[c, i, j], 0)
        i += 1


# test with some fixed upper and lower bounds
def verify_with_marabou_test(network:MarabouNetwork, image:np.array, label:int, box, occlusion_size=(1, 1), epsilon = 0.5):
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



