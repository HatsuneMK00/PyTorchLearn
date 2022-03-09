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
output_dim = 7
batch_num = 1
result_file_dir = '../experiment/results/'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


# thought #3 in the doc
def verify_occlusion_with_fixed_size(image: np.array, label: int, occlusion_size: tuple, occlusion_color: int):
    """
    given an image, label, occlusion size and occlusion color, verify that no matter where the occlusion area is
    network can classify correctly
    :param image: 1*3*32*32 image in np array after normalization
    :param label: int indicates the correct label
    :param occlusion_size: int indicates the occlusion size
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

    # define the constraints on the entire image
    # constraints = calculate_constrains(image, inputs)
    x = network.getNewVariable()
    y = network.getNewVariable()
    network.setLowerBound(x, 0)
    network.setUpperBound(x, w - 1)
    network.setLowerBound(y, 0)
    network.setUpperBound(y, h - 1)

    # iterate over the entire image
    for i in range(h):
        for j in range(w):
            # occlusion point cover (i, j)
            # the constraints should have size like [[eq1, eq2], [eq3, eq4], ...]
            # stand for (eq1 and eq1) or (eq2 and eq2) or ...
            # network.addDisjunctionConstraint(constraints)
            constraints = []
            eqs = []
            eq1 = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq1.addAddend(1, x)
            eq1.addScalar(j)
            eqs.append(eq1)
            eq2 = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq2.addAddend(1, y)
            eq2.addScalar(i)
            eqs.append(eq2)
            for k in range(c):
                eq3 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq3.addAddend(1, inputs[k][i][j])
                eq3.addScalar(occlusion_color)
                eqs.append(eq3)
            constraints.append(eqs)
            # otherwise
            # since don't know how to write unequal constraints
            # change two unequal constraints into four greater equal and less equal constraints
            # and also don't know if there exists greater and less relation
            eqs = []
            for k in range(c):
                eq8 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq8.addAddend(1, inputs[k][i][j])
                eq8.addScalar(image[k][i][j])
                eqs.append(eq8)
            eqs_temp = eqs.copy()
            eq4 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq4.addAddend(1, x)
            eq4.addScalar(j)
            eqs.append(eq4)
            constraints.append(eqs)
            eqs = eqs_temp.copy()
            eq5 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq5.addAddend(1, x)
            eq5.addScalar(j)
            eqs.append(eq5)
            constraints.append(eqs)
            eqs = eqs_temp.copy()
            eq6 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq6.addAddend(1, y)
            eq6.addScalar(i)
            eqs.append(eq6)
            constraints.append(eqs)
            eqs = eqs_temp.copy()
            eq7 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq7.addAddend(1, y)
            eq7.addScalar(i)
            eqs.append(eq7)
            constraints.append(eqs)
            # add constraints to network
            network.addDisjunctionConstraint(constraints)

    # add additional bounds for the inputs
    for i in range(h):
        for j in range(w):
            for k in range(c):
                network.setLowerBound(inputs[k, i, j], 0)
                network.setUpperBound(inputs[k, i, j], 1)

    # add bounds to outputs
    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs_flattened[i], outputs_flattened[label]], [1, -1], 0)
    constraints_calculation_end_time = time.monotonic()
    constraints_calculation_time = constraints_calculation_end_time - constraints_calculation_start_time

    verify_start_time = time.monotonic()
    vals, stats = network.solve(verbose=True)
    verify_end_time = time.monotonic()
    verify_time = verify_end_time - verify_start_time

    print("vals: ", vals)
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

    for i in range(batch_num):
        image, label = iterable_img_loader.next()
        image = image.numpy()
        label = label.item()

        for target_label in range(output_dim):
            if target_label == label:
                continue
            vals, constraints_calculation_time, verify_time = verify_occlusion_with_fixed_size(image, label, occlusion_size, occlusion_color)
            print('constraints_calculation_time: ', constraints_calculation_time)
            print('verify_time: ', verify_time)


