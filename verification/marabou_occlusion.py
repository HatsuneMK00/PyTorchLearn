# -*- coding: utf-8 -*-
# created by makise, 2022/3/1

# This script is used to verify occlusion type perturbation with Marabou.
import json

import onnx
import onnxruntime
from maraboupy import Marabou, MarabouNetwork
from PIL import Image
import numpy as np
import time

from marabou_utils import load_network, load_sample_image, get_test_images_loader
from occlusion_bound import calculate_entire_bounds

# define some global variables
model_name = "cnn_model_gtsrb_small.onnx"
occlusion_point = (16, 16)
occlusion_size = (5, 5)
occlusion_color = 0
epsilon = 0.5
input_size = (32, 32)
output_dim = 43
batch_num = 10
result_file_dir = '../experiment/results/'
# todo add time stamp to filename


# verify with marabou
def verify_with_marabou(image: np.array, label: int, box, occlusion_size,
                        occlusion_color, epsilon):
    bound_calculation_start_time = time.monotonic()
    # load network
    network = load_network(model_name)
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

    upper_bounds, lower_bounds = calculate_entire_bounds(image, left_upper_occ, occlusion_size,
                                                         occlusion_color, left_upper_affected,
                                                         (height_affected, width_affected), epsilon)

    # ------------------------------------------------------------------------------------------
    # set network input bounds according to lower_bounds, upper_bounds
    # ------------------------------------------------------------------------------------------
    # iterate over changed
    for i in range(h):
        for j in range(w):
            for channel in range(c):
                network.setUpperBound(inputs[channel][i][j], upper_bounds[i][j][channel])
                network.setLowerBound(inputs[channel][i][j], lower_bounds[i][j][channel])
    # ------------------------------------------------------------------------------------------
    # set bounds for network output
    # ------------------------------------------------------------------------------------------
    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs_flattened[i], outputs_flattened[label]], [1, -1], 0)

    bound_calculation_time = time.monotonic() - bound_calculation_start_time

    verify_start_time = time.monotonic()
    vals = network.solve(verbose=True)  # vals is a list, not a tuple as document says
    verify_time = time.monotonic() - verify_start_time
    print("vals length", len(vals))

    return vals, bound_calculation_time, verify_time


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
    img_loader = get_test_images_loader(input_size)
    iterable_img_loader = iter(img_loader)

    results = []
    # iterate first <batch_num> batch of images in img_loader
    for i in range(batch_num):
        start_time = time.monotonic()
        results_batch = []
        # get image and label
        image, label = iterable_img_loader.next()
        # convert tensor into numpy array
        image = image.numpy()
        label = label.item()
        isRobust = True
        bound_calculation_time = -1.0
        verify_time = -1.0
        predicted_label = -1
        results_batch = []
        adversarial_example = None
        for target_label in range(output_dim):
            if target_label == label:
                continue
            vals, bound_calculation_time, verify_time = verify_with_marabou(image, target_label,
                                                                            occlusion_point,
                                                                            occlusion_size, occlusion_color, epsilon)
            results_batch.append(
                {'vals': vals[0], 'bound_calculation_time': bound_calculation_time, 'verify_time': verify_time,
                 'target_label': target_label})
            # not robust in this target label
            if vals[0] == 'sat':
                adversarial_example = vals[1]
                predicted_label = target_label
                isRobust = False
                break
        # pack vals, bound_calculation_time, verify_time into a dict and append it to results
        total_time = time.monotonic() - start_time
        results.append(
            {'robust': isRobust, 'total_verify_time': total_time,
             'true_label:': label, 'predicted_label': predicted_label, 'adversarial_example': adversarial_example.tolist(),
             'origin_image': image.tolist(), 'detail': results_batch})

    # save results to file
    # encode model name, batch_num, occlusion_point, occlusion_size, occlusion_color, epsilon into filename
    result_filepath = result_file_dir + f'{model_name}_batchNum_{batch_num}_occlusionPoint_{occlusion_point[0]}_{occlusion_point[1]}_occlusionSize_{occlusion_size[0]}_{occlusion_size[1]}_occlusionColor_{occlusion_color}_epsilon_{epsilon}_outputDim_{output_dim}.json'
    with open(result_filepath, 'w') as f:
        json.dump(results, f)
        f.write('\n')
        f.flush()
