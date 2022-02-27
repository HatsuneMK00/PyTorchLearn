# -*- coding: utf-8 -*-
# created by makise, 2022/2/27

# This script is used to find the epsilon value with the Marabou solver.
# It's an attempt to find the epsilon value for the given network and how to use Marabou framework.


from maraboupy import Marabou, MarabouNetwork
import numpy as np
from PIL import Image

from marabou_utils import load_network


def find_epsilon(network:MarabouNetwork, image:np.array, label:int, epsilon:float):
    """
    :param network: MarabouNetwork
    :param image: input image in np array
    :param label: label of the image
    :param epsilon: the maximum epsilon value
    :return: the epsilon founded for this network
    """
    n_inputs = network.inputVars[0].flatten().shape[0]
    print("input_shape", network.inputVars[0].flatten().shape)
    n_outputs = network.outputVars[0].flatten().shape[0]
    print("output_shape", network.outputVars[0].flatten().shape)
    flattened_image = image.flatten()

    # get a marabou variable
    eps = network.getNewVariable()
    # set the upper bound and lower bound for this variable eps
    network.setLowerBound(eps, 0)
    network.setUpperBound(eps, epsilon)
    # set the eps as network inputVars
    network.inputVars = np.array([eps])

    # iterate through the inputs and set some equalities
    for i in range(n_inputs):
        val = flattened_image[i]
        network.addEquality([i, eps], [1, val - 1], val)

    # iterate through the outputs and set some equalities
    for i in range(n_outputs):
        if i != label:
            network.addEquality([network.outputVars[0][i], network.outputVars[0][label]], [1, -1], 0)

    vals, stats = network.solve(verbose=1)
    print("vals: ", vals)
    print("stats: ", stats)


if __name__ == '__main__':
    # load one image from training set using PIL and convert it to np array
    image = Image.open("../data/GTSRB/trainingset/00000/00000_00000.ppm")
     = np.array(image)
    print("np_img shape: ", np_img.shape)

    # load the network
    network = load_network('../model/fnn_model_gtsrb_small.onnx')
    label = 0
    epsilon = 0.5

    # conduct the find_epsilon function
    find_epsilon(network, np_img, label, epsilon)