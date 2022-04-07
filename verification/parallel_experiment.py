# -*- coding: utf-8 -*-
# created by makise, 2022/4/2

"""
This file is used to conduct experiments parallely to find out a better block size.
"""
import json
import os
import time
from multiprocessing.dummy import Pool

import numpy as np

from three_marabou_occlusion import conduct_experiment
from verification.show_adv_example import get_adv_examples

pe_result_dir = "./experiment/results/thought_3/pe_20220406_172949/"


def analyze_result():
    # load all file in the result directory
    ite = os.walk(pe_result_dir)
    results = {}
    for path, _, file_list in ite:
        for file_name in file_list:
            if file_name.endswith(".json"):
                with open(os.path.join(path, file_name), "r") as f:
                    # f is a json file
                    data = json.load(f)
                    adv_examples = get_adv_examples(data)
                    # fixme only use result of the first image
                    results[file_name] = {
                        "file_name": file_name,
                        "total_verify_time": adv_examples[0]['total_verify_time'],
                    }
    # sort the results by total_verify_time
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_verify_time'])

    # print the result
    print("------------------Result of {}------------------".format(pe_result_dir))
    for result in sorted_results:
        print("{} : {}".format(result['file_name'], result['total_verify_time']))
    print("------------------------------------------------")




# fixme Due to file I/O (probably), the parallelization is not working. Set processes to 1 on server currently.
if __name__ == '__main__':
    parameters = []
    block_sizes = [(8, 8), (16, 16), (32, 32)]
    occlusion_color = 0
    # occlusion size is a list of tuple (i, i), i ranges from 0 to 31
    occlusion_sizes = [(i, i) for i in range(2, 8)]

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

    for block_size in block_sizes:
        for occlusion_size in occlusion_sizes:
            parameters.append((occlusion_size, occlusion_color, block_size, timestamp))

    # conduct experiments parallely
    pool = Pool(processes=1)
    results = pool.starmap(conduct_experiment, parameters)
    pool.close()
    pool.join(timeout=60 * 60 * 8)

    print("Parallel Experiment: Done!")
