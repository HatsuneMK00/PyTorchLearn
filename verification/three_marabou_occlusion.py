# -*- coding: utf-8 -*-
# created by makise, 2022/3/8

# corresponding to the thought #3 in the doc
# given occlusion color, given occlusion size, verify that no matter where the occlusion area is
# network can classify correctly

import json

import onnx
import onnxruntime
from maraboupy import Marabou, MarabouNetwork, MarabouCore
from PIL import Image
import numpy as np
import time

from marabou_utils import load_network, load_sample_image, get_test_images_loader
from occlusion_bound import calculate_entire_bounds

# define some global variables
model_name = "fnn_model_gtsrb_small.onnx"
occlusion_size = (1, 1)
occlusion_color = 0
input_size = (32, 32)
channel = 3
output_dim = 7
batch_num = 1
result_file_dir = '../experiment/results/thought_3/'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

mean, std = np.array([0.3337, 0.3064, 0.3171]), np.array([0.2672, 0.2564, 0.2629])
epsilon = 1e-6

# thought #3 in the doc
def verify_occlusion_with_fixed_size(image: np.array, label: int, occlusion_size: tuple, occlusion_color: int):
    """
    given an image, label, occlusion size and occlusion color, verify that no matter where the occlusion area is
    network can classify correctly
    :param image: 1*3*32*32 image in np array after normalization
    :param label: int indicates the correct label
    :param occlusion_size: tuple indicates the occlusion size
    :param occlusion_color: int indicates the occlusion color
    :return: vals, constraints_calculation_time, verify_time
    """
    constraints_calculation_start_time = time.monotonic()
    # load network
    network = load_network(model_name)
    inputs = network.inputVars[0][0] # 3*32*32
    outputs = network.outputVars[0] # {output_dim}
    n_outputs = outputs.flatten().shape[0]
    outputs_flattened = outputs.flatten()
    image = image[0] # 3*32*32
    assert image.shape == inputs.shape
    assert n_outputs == output_dim

    c, h, w = image.shape
    occlusion_height, occlusion_width = occlusion_size

    # define the constraints on the entire image
    # constraints = calculate_constrains(image, inputs)
    x = network.getNewVariable()
    y = network.getNewVariable()
    network.setLowerBound(x, 0)
    network.setUpperBound(x, w - occlusion_width)
    network.setLowerBound(y, 0)
    network.setUpperBound(y, h - occlusion_height)

    # iterate over the entire image
    for i in range(h):
        for j in range(w):
            # occlusion point cover (i, j)
            # the constraints should have size like [[eq1, eq2], [eq3, eq4], ...]
            # stand for (eq1 and eq1) or (eq2 and eq2) or ...
            # network.addDisjunctionConstraint(constraints)
            constraints = []
            # this equation is like (x <= j) and (x >= j - occlusion_size[0] - 1) and (y <= i) and
            # (y >= i - occlusion_size[1] - 1) and (image[i, j] == occlusion_color)
            # with the simple occlusion_size = (1, 1), inequality becomes equality
            eqs = []
            eq1 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq1.addAddend(1, x)
            eq1.setScalar(j + 1)
            eqs.append(eq1)
            eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq2.addAddend(1, x)
            eq2.setScalar(j - occlusion_width + 1)
            eqs.append(eq2)
            eq3 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq3.addAddend(1, y)
            eq3.setScalar(i + 1)
            eqs.append(eq3)
            eq4 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq4.addAddend(1, y)
            eq4.setScalar(i - occlusion_height + 1)
            eqs.append(eq4)
            for k in range(c):
                eq5 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq5.addAddend(1, inputs[k][i][j])
                eq5.setScalar(((occlusion_color / 255.0) - mean[k]) / std[k])
                eqs.append(eq5)
            constraints.append(eqs)
            # otherwise
            # since don't know how to write unequal constraints
            # change two unequal constraints into four greater equal and less equal constraints
            # and also don't know if there exists greater and less relation
            # this equation is like (x >= j) and image[i, j] == origin_color
            # this has four similar constraints connecting with or relation
            # the other three has the same second part and the first part is separately
            # (x <= j - occlusion_size[0] - 1) and (j >= i) and (y <= i - occlusion_size[1] - 1)
            eqs = []
            for k in range(c):
                eq10 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq10.addAddend(1, inputs[k][i][j])
                eq10.setScalar(image[k][i][j])
                eqs.append(eq10)
            eqs_temp = eqs.copy()
            eq6 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq6.addAddend(1, x)
            eq6.setScalar(j + epsilon)
            eqs.append(eq6)
            constraints.append(eqs)
            eqs = eqs_temp.copy()
            eq7 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq7.addAddend(1, x)
            eq7.setScalar(j - occlusion_width + 1 - epsilon)
            eqs.append(eq7)
            constraints.append(eqs)
            eqs = eqs_temp.copy()
            eq8 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq8.addAddend(1, y)
            eq8.setScalar(i + epsilon)
            eqs.append(eq8)
            constraints.append(eqs)
            eqs = eqs_temp.copy()
            eq9 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq9.addAddend(1, y)
            eq9.setScalar(i - occlusion_height + 1 - epsilon)
            eqs.append(eq9)
            constraints.append(eqs)
            # add constraints to network
            network.addDisjunctionConstraint(constraints)

    lower_bound = (0 - mean) / std
    upper_bound = (1 - mean) / std
    # add additional bounds for the inputs
    for i in range(h):
        for j in range(w):
            for k in range(c):
                network.setLowerBound(inputs[k, i, j], lower_bound[k])
                network.setUpperBound(inputs[k, i, j], upper_bound[k])

    # add bounds to outputs
    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs_flattened[i], outputs_flattened[label]], [1, -1], 0)
    constraints_calculation_end_time = time.monotonic()
    constraints_calculation_time = constraints_calculation_end_time - constraints_calculation_start_time

    verify_start_time = time.monotonic()
    vals = network.solve(verbose=True)
    verify_end_time = time.monotonic()
    verify_time = verify_end_time - verify_start_time

    print("vals length: ", len(vals))

    return vals, constraints_calculation_time, verify_time


def calculate_constrains(image, inputs):
    """
    calculate the constraints for the entire image
    :return: constraints
    """
    c, h, w = image.shape
    constraints = []
    test_constraints = []
    for i in range(h):
        for j in range(w):
            eqs = []
            test_eqs = []
            for k in range(c):
                eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq.addAddend(1, inputs[k, i, j])
                eq.setScalar(occlusion_color)
                eqs.append(eq)
                test_eq = f'x({k}, {i}, {j}) = {occlusion_color}'
                test_eqs.append(test_eq)
            for ii in range(h):
                for jj in range(w):
                    if i == ii and j == jj:
                        continue
                    for kk in range(c):
                        eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq.addAddend(1, inputs[kk, ii, jj])
                        eq.setScalar(image[kk, ii, jj])
                        eqs.append(eq)
                        test_eq = f'x({kk}, {ii}, {jj}) = {image[kk, ii, jj]}'
                        test_eqs.append(test_eq)
            constraints.append(eqs)
            test_constraints.append(test_eqs)
    print("test_constraints length: ", len(test_constraints))
    print("test_constraints[0] length: ", len(test_constraints[0]))
    return constraints


if __name__ == '__main__':
    img_loader = get_test_images_loader(input_size, output_dim=output_dim)
    iterable_img_loader = iter(img_loader)

    results = []
    for i in range(batch_num):
        start_time = time.monotonic()
        image, label = iterable_img_loader.next()
        image = image.numpy()
        label = label.item()
        isRobust = True
        constraints_calculation_time = -1.0
        verify_time = -1.0
        predicted_label = -1
        results_batch = []
        adversarial_example = None
        adv_example_list = None
        for target_label in range(output_dim):
            if target_label == label:
                continue
            vals, constraints_calculation_time, verify_time = verify_occlusion_with_fixed_size(image, target_label, occlusion_size, occlusion_color)
            results_batch.append(
                {'vals': vals[0], 'constraints_calculation_time': constraints_calculation_time, 'verify_time': verify_time,
                 'target_label': target_label})
            if vals[0] == 'sat':
                adversarial_example = vals[1]
                # unpack adversarial example to a list
                # adversarial_example is a dict{int, float}
                # key is the index of the variable in the network
                # value is the value of the variable
                adv_example_list = [adversarial_example[i] for i in range(channel * input_size[0] * input_size[1])]
                predicted_label = target_label
                isRobust = False
                break
        total_time = time.monotonic() - start_time

        results.append(
            {'robust': isRobust, 'total_verify_time': total_time,
             'true_label:': label, 'predicted_label': predicted_label, 'adv_example': adv_example_list,
             'origin_image': image.tolist(), 'detail': results_batch})

    # save results to file
    result_filepath = result_file_dir + f'{model_name}_batchNum_{batch_num}_occlusionSize_{occlusion_size[0]}_{occlusion_size[1]}_occlusionColor_{occlusion_color}_outputDim_{output_dim}_{timestamp}.json'
    with open(result_filepath, 'w') as f:
        json.dump(results, f)
        f.write('\n')
        f.flush()



