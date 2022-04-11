# -*- coding: utf-8 -*-
# created by makise, 2022/3/7

# read result file produced from marabou_occlusion verification and show the adversarial example
# together with the original image
import json

import numpy as np
from matplotlib import pyplot as plt

result_file_dir = '/home/GuoXingWu/occlusion_veri/PyTorchLearn/experiment/results/thought_4/pe_20220411_160905/'
result_filename = 'cnn_model_gtsrb_small_2.onnx_batchNum_1_occlusionSize_20_20_colorEpsilon_0.01_outputDim_7_blockSize_32_32.json'


def read_result_file(filename):
    with open(filename, 'r') as f:
        # read result file as json
        result = json.load(f)

    return result


def get_adv_examples(result):
    # get the adversarial example
    adv_examples = []
    # get occlusion size from result filename
    occlusion_height = result_filename.split('_')[7]
    occlusion_width = result_filename.split('_')[8]
    for i in range(len(result)):
        predicted_label = result[i]['predicted_label']
        total_verify_time = result[i]['total_verify_time']
        true_label = result[i]['true_label']
        origin_image = result[i]['origin_image']  # size is (1, 32, 32, 3)
        idx = i
        # pack them into dict and append to adv_examples
        adv_example_dict = {'predicted_label': predicted_label,
                            'true_label': true_label,
                            'origin_image': origin_image,
                            'idx': idx,
                            'occlusion_size': (occlusion_height, occlusion_width),
                            'total_verify_time': total_verify_time}
        if not result[i]['robust']:
            adv_example = result[i]['adv_example']  # size is (32 * 32 * 3, )
            adv_example_dict['adv_example'] = adv_example
        adv_examples.append(adv_example_dict)
    return adv_examples


def show_adv_example(adv_examples, show = range(1)):
    """
    Show the adversarial example together with the original image using plt
    Only show adversarial example in the range defined by show
    :param adv_examples: parsed result from result file
    :param show: range of the adversarial example to show, default is range(5)
    :return:
    """
    # convert adv_examples and origin image to numpy array
    assert len(adv_examples) >= len(show)
    for i in show:
        adv_example = np.array(adv_examples[i]['adv_example'])
        adv_example = np.reshape(adv_example, (3, 32, 32))
        adv_example = np.transpose(adv_example, (1, 2, 0))
        adv_example = denormalize_image(adv_example)
        origin_image = np.array(adv_examples[i]['origin_image'])
        origin_image = np.reshape(origin_image, (3, 32, 32))
        origin_image = np.transpose(origin_image, (1, 2, 0))
        origin_image = denormalize_image(origin_image)
        predicted_label = adv_examples[i]['predicted_label']
        true_label = adv_examples[i]['true_label']
        idx = adv_examples[i]['idx']
        occlusion_height, occlusion_width = adv_examples[i]['occlusion_size']
        # show the adversarial example
        plt.subplot(1, 2, 1)
        plt.imshow(adv_example)
        plt.title('adversarial example')
        plt.subplot(1, 2, 2)
        plt.imshow(origin_image)
        plt.title('origin image')
        plt.suptitle('predicted label: %d, true label: %d, idx: %d, occlusion size: (%s, %s)' % (predicted_label, true_label, idx, occlusion_height, occlusion_width))
        plt.show()
        # count the number of different pixels
        diff_pixels = 0
        diff_pixels_pos = []
        diff_pixels_value = []
        for i in range(32):
            for j in range(32):
                for k in range(3):
                    if adv_example[i][j][k] != origin_image[i][j][k]:
                        diff_pixels += 1
                        diff_pixels_pos.append((i, j, k))
                        diff_pixels_value.append(adv_example[i][j][k] - origin_image[i][j][k])
        print("different pixels: ", diff_pixels)
        print("different pixels position: ", diff_pixels_pos)
        print("different pixels value: ", diff_pixels_value)

def denormalize_image(image):
    """
    Denormalize the image to [0, 255]
    :param image: image  of size (32, 32, 3)
    :return: image after denormalization
    """
    mean, std = np.array([0.3337, 0.3064, 0.3171]), np.array([0.2672, 0.2564, 0.2629])
    image = image * std + mean
    return image


if __name__ == '__main__':
    path = result_file_dir + result_filename
    result = read_result_file(path)
    adv_examples = get_adv_examples(result)
    show_adv_example(adv_examples)
