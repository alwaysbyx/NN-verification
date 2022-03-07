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
import utils
import pandas as pd
#import matlab.engine



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', '../models/nips_pgd.ckpt',
                    'Path of checkpoint with trained model to verify')
flags.DEFINE_string('model_json', '../model_details/nips_pgd.json',
                    'Path of json file with model description')
flags.DEFINE_string('test_input', '../mnist_permuted_data/test-0.npy',
                    'Path of numpy file with test input to certify')
flags.DEFINE_integer('true_class', 8,
                     'True class of the test input')
flags.DEFINE_integer('adv_class', 0,
                     'target class of adversarial example; test all classes if -1')
flags.DEFINE_integer('num_classes', 10,
                     'total number of classes to verify against')
flags.DEFINE_float('input_minval', 0,
                   'Minimum value of valid input')
flags.DEFINE_float('input_maxval', 1,
                   'Maximum value of valid input')
flags.DEFINE_float('epsilon', 0.1,
                   'Size of perturbation')
# Working folder to save the .m files that the matlab function reads 
flags.DEFINE_string('matlab_folder', 'matlab',
                     'Folder to save matlab things')
flags.DEFINE_integer('input_dimension', 784,
                    'Folder to save matlab things')

dataset = 'MNIST'
MATLAB_PATH = "/Applications/MATLAB_R2019b.app/bin/matlab"
def main(_):

  # Reading test input and reshaping
  # with tf.gfile.Open(FLAGS.test_input) as f:
  #   test_input = np.load(f)
  test_input = np.load(FLAGS.test_input)
  if(dataset == 'MNIST'):
    num_rows = 28
    num_columns = 28
    num_channels = 1

  print("Running certification for input file", FLAGS.test_input)
  net_weights, net_biases, net_layer_types = read_weights.read_weights(
      FLAGS.checkpoint, FLAGS.model_json, [num_rows, num_columns, num_channels])
  # If want to use a random network 
  # net_weights, net_biases, net_layer_types = utils.create_random_network(
  # FLAGS.input_dimension, FLAGS.num_layers, FLAGS.layer_width, FLAGS.num_classes)

  nn_params = neural_net_params.NeuralNetParams(
      net_weights, net_biases, net_layer_types)

  test_input = np.reshape(test_input, [-1, 1])
  
  if FLAGS.adv_class == -1:
    start_class = 0
    end_class = FLAGS.num_classes
  else:
    start_class = FLAGS.adv_class
    end_class = FLAGS.adv_class + 1
  for adv_class in range(start_class, end_class):
    print('Adv class', adv_class)
    if adv_class == FLAGS.true_class:
      continue

    config = tf.ConfigProto(
         device_count = {'GPU': 0})

    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      if not os.path.exists(FLAGS.matlab_folder):
        os.mkdir(FLAGS.matlab_folder)
      matlab_object = matlab_interface.MatlabInterface(FLAGS.matlab_folder)
      matlab_object.save_weights(nn_params, sess)
      opt_params = {}
      opt_params['test_input'] = test_input 
      opt_params['epsilon'] = FLAGS.epsilon 
      opt_params['true_class'] = FLAGS.true_class 
      opt_params['adv_class'] = adv_class
      opt_params['final_linear'] = sess.run(nn_params.final_weights[adv_class, :]
                         - nn_params.final_weights[FLAGS.true_class, :])
      opt_params['final_constant'] = sess.run(nn_params.final_bias[adv_class]
                           - nn_params.final_bias[FLAGS.true_class])
      lower, upper = compute_bounds.compute_bounds(test_input, FLAGS.epsilon, 
                                                   FLAGS.input_minval, FLAGS.input_maxval, nn_params)
      opt_params['lower'] = [sess.run(l) for l in lower]
      opt_params['upper'] = [sess.run(u) for u in upper]
      
      matlab_object.save_opt_params(opt_params)
      break
      command_string =  MATLAB_PATH + " -nodisplay -r \"matlab_lp(\'" + FLAGS.matlab_folder + "\')\""
      print('following command',command_string)
      os.system(command_string)
      # eng = matlab.engine.start_matlab()
      # eng.matlab_sdp(FLAGS.matlab_folder)
      opt_val = sio.loadmat(os.path.join(FLAGS.matlab_folder, 'SDP_optimum.mat'))
      val = opt_val['val'][0][0]
      time = opt_val['time'][0][0]
      df = pd.read_csv('../result/LP-LP.csv', index_col=0)
      df.loc[len(df)] = [FLAGS.test_input[-10:], val, time, FLAGS.adv_class, FLAGS.true_class]
      df.to_csv('../result/LP-LP.csv')
      if(val < 0):
        print('Input example is robust to perturbation to adv class ' + str(adv_class))
      else:
        print('Input example cannot be certified as robust to perturbation to adv class ' + str(adv_class))
        exit()
    print('Input example succesfully verified')

if __name__ == '__main__':
  tf.app.run(main)
