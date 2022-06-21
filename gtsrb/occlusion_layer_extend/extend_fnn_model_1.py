# -*- coding: utf-8 -*-
# created by makise, 2022/6/21

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image

from gtsrb.fnn_model_1 import SmallDNNModel
from gtsrb.occlusion_layer_extend.occlusion_layer import OcclusionLayer

def get_a_test_image():
    image = Image.open("../../data/GTSRB/trainingset/00000/00000_00000.ppm")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ])
    tensor = transform(image)
    return tensor


if __name__ == '__main__':
    # load pytorch model from model/fnn_model_gtsrb_small.pth
    model = SmallDNNModel()
    model.load_state_dict(torch.load('../../model/fnn_model_gtsrb_small.pth', map_location=torch.device('cpu')))
    # generate random input of size (3, 32, 32), type is torch.Tensor
    input = torch.Tensor([1, 1])
    fake_image = torch.ones(3, 32, 32)
    image = get_a_test_image()
    # add a new custom layer OcclusionLayer in front of the model
    occlusion_layer = OcclusionLayer(image=image)
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())
    )
    # print the summary of the extended model
    # print(extended_model)

    output = extended_model(input)
    output_origin = model(image)
    print(output)
    print(output_origin)

    # occluded_image = occlusion_layer(input)
    # plt.subplot(1, 2, 1)
    # # convert torch into numpy array
    # occluded_image = occluded_image.numpy()
    # occluded_image = occluded_image.reshape(3, 32, 32).transpose(1, 2, 0)
    # plt.imshow(occluded_image)
    # plt.subplot(1, 2, 2)
    # image = image.numpy()
    # image = image.reshape(3, 32, 32).transpose(1, 2, 0)
    # plt.imshow(image)
    # plt.show()



