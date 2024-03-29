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
    image = image.reshape(3, 32, 32)
    occlusion_layer_v4 = OcclusionLayerV4(image, None, False)
    input_tensor = torch.tensor(input)
    epsilons = torch.tensor(epsilons)
    occluded_image = occlusion_layer_v4(input_tensor, epsilons)
    # convert torch into numpy array
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(3, 32, 32)
    occluded_image = occluded_image.transpose(1, 2, 0)
    # clip occluded_image into [0, 1] with numpy
    occluded_image = np.clip(occluded_image, 0, 1)
    plt.imsave('multiform_occluded_gtsrb_18.png', occluded_image)
    # set image type as np.float32
    image = image.numpy()
    image = image.reshape(3, 32, 32)
    image = image.transpose(1, 2, 0)
    plt.imsave('multiform_original_gtsrb_18.png', image)
    # show origin and occluded image
    # plt.subplot(1, 2, 1)
    # plt.imshow(occluded_image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(image)
    # plt.show()




def show_occluded_image_v3(image, input, color):
    image = image.numpy()
    image = image.reshape(3, 32, 32)
    image = image.transpose(1, 2, 0)

    occluded_image = occlusion.occlusion_with_interpolation(image, (input[2], input[0]), (input[1], input[3]), color)

    # plt.imshow(occluded_image)
    # plt.show()

    plt.imsave('uniform_occluded_gtsrb_16.png', np.clip(occluded_image, 0, 1))
    plt.imsave('uniform_original_gtsrb_16.png', image)

    # image_reshape = image.reshape(3, 32, 32)
    # occlusion_layer = OcclusionLayer(image=image_reshape, first_layer=None, is_cnn=True)
    # input_tensor = torch.tensor([*input, color])
    # occluded_image = occlusion_layer(input_tensor)
    # plt.subplot(1, 2, 1)
    # convert torch into numpy array
    # occluded_image = occluded_image.numpy()
    # occluded_image = occluded_image.reshape(3, 32, 32)
    # occluded_image = occluded_image.transpose(1, 2, 0)
    # plt.imshow(occluded_image)
    # plt.subplot(1, 2, 2)
    # set image type as np.float32
    # image = image.numpy()
    # image = image.reshape(3, 32, 32)
    # image = image.transpose(1, 2, 0)
    # plt.imshow(image)
    # plt.savefig(fname='v3_example.png', format='png')
    # plt.show()


def get_sample_image(idx):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])
    test_data = GTSRB(root_dir='../../data/', train=False, transform=transform, classes=[1, 2, 3, 4, 5, 7, 8])
    test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    iter_on_loader = iter(test_loader)

    # get image at index idx
    for i in range(idx + 1):
        image, label = next(iter_on_loader)
    return image, label


def show_gtsrb_data_in_grid():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])
    test_data = GTSRB(root_dir='../../data/', train=False, transform=transform, classes=[1, 2, 3, 4, 5, 7, 8])
    test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    iter_on_loader = iter(test_loader)

    # get image at index idx
    for i in range(100):
        image, label = next(iter_on_loader)
        image = image.numpy()
        image = image.reshape(3, 32, 32)
        image = image.transpose(1, 2, 0)
        plt.subplot(10, 10, i + 1)
        # remove all ticks
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
    plt.show()


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
    # image, label = get_sample_image(4)
    # print("label: ", label.item())
    # v4_layer_input = []
    # epsilons = []
    # # read json object from ./result4.json
    # with open('result4_0.4_sort_1_fnn3_size2_for_example.json', 'r') as f:
    #     verification_results = json.load(f)
    #     for verification_result in verification_results:
    #         if verification_result['robust'] != False:
    #             continue
    #         adversarial_example = verification_result['adversarial_example']
    #         layer_input = [adversarial_example['a'], adversarial_example['size_a'], adversarial_example['b'], adversarial_example['size_b']]
    #         layer_epsilons = adversarial_example['epsilons']
    #         v4_layer_input.append(layer_input)
    #         epsilons.append(layer_epsilons)
    #         # print(layer_input)
    #         # print(layer_epsilons)
    #
    # images = [12, 3, 4, 14, 15, 18, 19, 26, 28]
    # # for i in range(len(v4_layer_input)):
    # i = 5
    # image, label = get_sample_image(images[i])
    # print(v4_layer_input[i])
    # show_occluded_image_v4(image, v4_layer_input[i], epsilons[i])


    # v3_layer_inputs = [
    #     [8.0, 3.0, 10.0, 3.0],
    #     [11.0, 3.0, 9.999999999999996, 3.0],
    #     [11.0, 3.0, 11.4, 3.0],
    #     [10.0, 3.0, 13.6, 3.0],
    #     [13.0, 3.0, 10.99999999999999, 3.0]
    # ]
    # colors = [0, 0.2, 0.4, 0.6000000000000001, 0.8]
    image, label = get_sample_image(6)
    print("label: ", label.item())
    show_occluded_image_v3(image, [13.0, 3.0, 11.0, 3.0], [0.0, 0.0, 1.0])
    # show_occluded_image_v3(image, [15.0, 2.0, 12.0, 2.0], [0, 0, 0])
    # show_gtsrb_data_in_grid()