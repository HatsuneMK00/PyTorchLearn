# -*- coding: utf-8 -*-
# created by makise, 2022/2/27

# Some useful functions for Marabou

from maraboupy import Marabou, MarabouNetwork
from PIL import Image
import numpy as np

# load a network from onnx format file and return MarabouNetwork
def load_network(filename) -> MarabouNetwork:
    """
    Load a network from onnx format file and return MarabouNetwork
    :param filename: the name of the onnx file
    :return: network: MarabouNetwork
    """
    # the path of network file
    path = '../model/' + filename
    # load the network
    network = Marabou.read_onnx(path)
    return network


def load_sample_image() -> np.ndarray:
    """
    Load one sample image and resize it to 32*32.
    :return: numpy array of image
    """
    # load one image from training set using PIL and convert it to np array
    image = Image.open("../data/GTSRB/trainingset/00000/00000_00000.ppm")
    # resize the image to (32, 32), if it's less than 32, pad it with black pixels
    image = image.resize((32, 32), Image.ANTIALIAS)
    np_img = np.array(image)
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np.reshape(np_img, (1, 3, 32, 32))
    return np_img