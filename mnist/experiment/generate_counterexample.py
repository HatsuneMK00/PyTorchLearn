# -*- coding: utf-8 -*-
# created by makise, 2022/9/5
import numpy as np
from matplotlib.patches import Rectangle

from occlusion_layer.occlusion_layer_v3 import OcclusionLayer as OcclusionLayerV3
from occlusion_layer.occlusion_layer_v4 import OcclusionLayer as OcclusionLayerV4
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
import torch.utils.data as data
from gtsrb.gtsrb_dataset import GTSRB
from interpolation import occlusion
import json

def show_occluded_image_v4(image, input, epsilons):
    image = image.reshape(1, 28, 28)
    occlusion_layer_v4 = OcclusionLayerV4(image, None, False)
    input_tensor = torch.tensor(input)
    epsilons = torch.tensor(epsilons)
    occluded_image = occlusion_layer_v4(input_tensor, epsilons)
    # convert torch into numpy array
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(28, 28)
    # clip occluded_image into [0, 1] with numpy
    occluded_image = np.clip(occluded_image, 0, 1)
    plt.imsave('multiform_occluded_mnist_1.png', occluded_image, cmap='gray')
    # set image type as np.float32
    image = image.numpy()
    image = image.reshape(28, 28)
    plt.imsave('multiform_original_mnist_1.png', image, cmap='gray')
    # show origin and occluded image
    # plt.subplot(1, 2, 1)
    # plt.imshow(occluded_image, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(image, cmap='gray')
    # plt.show()


def show_occluded_image_v3(image, input, color):
    # image = image.numpy()
    image = image.reshape(1, 28, 28)

    occlusion_layer_v3 = OcclusionLayerV3(image, None, True)
    input_tensor = torch.tensor([*input, color])
    occluded_image = occlusion_layer_v3(input_tensor)
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(28, 28)
    occluded_image = np.clip(occluded_image, 0, 1)

    # plt.imshow(occluded_image, cmap='gray')
    # plt.show()

    # save gray image
    plt.imsave('uniform_occluded_mnist_12.png', occluded_image, cmap='gray')
    plt.imsave('uniform_original_mnist_12.png', image.numpy().reshape(28, 28), cmap='gray')


def get_sample_image(idx):
    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_on_loader = iter(test_loader)

    # get image at index idx
    for i in range(idx + 1):
        image, label = next(iter_on_loader)
    return image, label


def show_mnist_data_in_grid():
    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_on_loader = iter(test_loader)

    for i in range(100):
        image, label = next(iter_on_loader)
        image = image.numpy()
        image = image.reshape(28, 28)
        plt.subplot(10, 10, i + 1)
        # remove all ticks
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    # i = 4
    # image, label = get_sample_image(17)
    # print("label: ", label.item())
    v4_layer_input = []
    epsilons = []
    # read json object from ./result4.json
    # with open('./result4_fnn2_0.4_sort_1_size2_for_example.json', 'r') as f:
    #     verification_results = json.load(f)
    #     # only consider the image #4
    #     for verification_result in verification_results:
    #         if verification_result['robust'] != False:
    #             continue
    #         adversarial_example = verification_result['adversarial_example']
    #         layer_input = [adversarial_example['a'], adversarial_example['size_a'], adversarial_example['b'], adversarial_example['size_b']]
    #         layer_epsilons = adversarial_example['epsilons']
    #         v4_layer_input.append(layer_input)
    #         epsilons.append(layer_epsilons)
    #
    # images = [0, 1, 6, 10, 17, 26]
    # # for i in range(len(v4_layer_input)):
    # i = 1
    # image, label = get_sample_image(images[i])
    # print("layer input: ", v4_layer_input[i])
    # show_occluded_image_v4(image, v4_layer_input[i], epsilons[i])


    # v3_layer_inputs = [
    #     [8.0, 3.0, 10.0, 3.0],
    #     [11.0, 3.0, 9.999999999999996, 3.0],
    #     [11.0, 3.0, 11.4, 3.0],
    #     [10.0, 3.0, 13.6, 3.0],
    #     [13.0, 3.0, 10.99999999999999, 3.0]
    # ]
    # colors = [0, 0.2, 0.4, 0.6000000000000001, 0.8]
    image, label = get_sample_image(12)
    print("label: ", label.item())
    show_occluded_image_v3(image, [14.0, 4.0, 16.0, 4.0], 0)
    # show_gtsrb_data_in_grid()
    # show_mnist_data_in_grid()