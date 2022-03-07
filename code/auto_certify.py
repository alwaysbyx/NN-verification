"""Code for running the certification problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.io as sio
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import neural_net_params
import read_weights
import matlab_interface
import compute_bounds 
import pandas as pd
import utils
#import matlab.engine



if __name__ == '__main__':
    labels = [8, 8, 1, 1, 3, 3, 5, 0, 2, 5, 9, 1, 2, 5, 0, 2, 2, 0, 1, 0, 4, 3, 4, 7, 8]
    network = 'nips_lp'
    for i in range(0,20):
        for adv_class in range(10):
            if adv_class == 0 or adv_class == labels[i]:
                continue
            # if i==3 and adv_class == 2:
            #     continue
            command = f"python certify.py --checkpoint ../models/{network}.ckpt --model_json ../model_details/{network}.json --test_input ../mnist_permuted_data/test-{i}.npy --true_class {labels[i]} --adv_class {adv_class} --epsilon 0.1 --matlab_folder matlab"
            print(command)
            os.system(command)
            opt_val = sio.loadmat(os.path.join('matlab', 'SDP_optimum.mat'))
            val = opt_val['val']
            if val >= 0:
                break
    # time = opt_val['time']
    # print(val, time)
    # df = pd.DataFrame(columns=['test_sample','verified','time', 'adv', 'true'])
    # df.to_csv('../result/LP-LP.csv')
        