# -*- coding: utf-8 -*-
# created by makise, 2022/4/2

"""
This file is used to conduct experiments parallely to find out a better block size.
"""

import time
from multiprocessing.dummy import Pool

import numpy as np

from three_marabou_occlusion import conduct_experiment

if __name__ == '__main__':
    parameters = []
    block_sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
    occlusion_color = 0
    # occlusion size is a list of tuple (i, i), i ranges from 0 to 31
    occlusion_sizes = [(i, i) for i in range(32)]

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

    for block_size in block_sizes:
        for occlusion_size in occlusion_sizes:
            parameters.append((occlusion_size, occlusion_color, block_size, timestamp))

    # conduct experiments parallely
    pool = Pool(processes=32)
    results = pool.map(conduct_experiment, parameters)
    pool.close()
    pool.join(timeout=7200)

    # save results
    np.save('pe_results.npy', results)