"""Code with matlab interface for the current problem"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np 
import scipy.io as sio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import neural_net_params 


flags = tf.app.flags
FLAGS = flags.FLAGS

class MatlabInterface(object):
    """ Class to handle matlab interfacing for debugging certification"""

    def __init__(self, matlab_folder):
        self.folder = matlab_folder
        print(self.folder)
        if not tf.gfile.IsDirectory(matlab_folder):
            tf.gfile.MkDir(matlab_folder)
    
    def save_weights(self, nn_params, sess):
        numpy_weights = [np.matrix(sess.run(w)) for w in nn_params.weights]
        numpy_biases = [sess.run(b) for b in nn_params.biases]
        numpy_net_sizes = nn_params.sizes
        for i in range(len(numpy_weights)):
            print(np.shape(numpy_weights[i]))

        sio.savemat(os.path.join(self.folder, 'biases.mat'), {'biases':numpy_biases})
        sio.savemat(os.path.join(self.folder, 'weights.mat'), {'weights':numpy_weights})
        sio.savemat(os.path.join(self.folder, 'sizes.mat'), {'sizes':numpy_net_sizes})
        
    def save_opt_params(self, opt_params):
        sio.savemat(os.path.join(self.folder, 'opt_params.mat'), {'opt_params':opt_params})
        
        
