# -*- coding: utf-8 -*-
# created by makise, 2022/2/23

"""
Use Marabou to conduct formal verification on a small CNN model on GTSRB dataset
"""

# use Marabou to read onnx format model
from maraboupy import Marabou
import numpy as np

# load onnx model
from torch.utils import data
from torchvision import transforms

from gtsrb.gtsrb_dataset import GTSRB

model_path = "../models/gtsrb/gtsrb_model_small.onnx"

# define the same data transform as when the model is trained
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])

# define the test data
test_data = GTSRB(root_dir='../data', train=False, transform=data_transform, classes=range(0, 7))
# create data loader for evaluating
test_loader = data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)
samples, labels = iter(test_loader).next()
print(samples.shape, labels.shape)

def load_model(model_path) -> Marabou.MarabouNetwork:
    """
    load onnx model
    :param model_path: path of onnx model
    :return:
    """
    # load onnx model
    m = Marabou.read_onnx(model_path)
    return m


# use marabou's evaluate method to evaluate the model
def evaluate_model(m: Marabou.MarabouNetwork, input_data, output_data):
    """
    evaluate the model
    :param m: marabou network
    :param input_data: input data
    :param output_data: output data
    :return:
    """
    # evaluate the model
    # iterate over all the input data
    result = []

    # use the first input_data to test the evaluating of the model
    test_input = input_data[0]
    print(test_input.shape)
    # create marabou options
    options = Marabou.createOptions(verbosity = 0)
    # evaluate the model with marabou using test_input and options
    marabou_output = m.evaluate(test_input, options=options)
    # print the output
    print(marabou_output)

    # for i in range(len(input_data)):
    #     marabou_output = m.evaluate(input_data[i], )
    #     print("marabou_output size: ", marabou_output.shape)
    #     if np.argmax(marabou_output[0]) == np.argmax(output_data[i]):
    #         result.append(i)
    #
    # acc = len(result) / len(input_data)
    # print("Accuracy evaluated by Marabou: ", acc)


if __name__ == '__main__':
    # load onnx model
    m = load_model(model_path)

    # evaluate the model
    input_data, output_data = samples, labels
    evaluate_model(m, input_data, output_data)