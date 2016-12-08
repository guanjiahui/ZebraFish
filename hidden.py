from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tensorflow as tf
import os.path

import pylab
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("file_name", "", "Checkpoint filename")
tf.app.flags.DEFINE_string("tensor_name", "", "Name of the tensor to inspect")
tf.app.flags.DEFINE_bool("all_tensors", "False",
                         "If True, print the values of all the tensors.")
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

def print_tensors_in_checkpoint_file(file_name, tensor_name):
  """Prints tensors in a checkpoint file.

  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.

  If `tensor_name` is provided, prints the content of the tensor.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
  """
  try:
    reader = tf.train.NewCheckpointReader(file_name)
    if FLAGS.all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name) 
      x = reader.get_tensor(tensor_name)
      init = tf.initialize_all_variables()
      sess = tf.InteractiveSession()
      sess.run(init)
      print(x)
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")



def main(unused_argv):
  if not FLAGS.file_name:
    print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
          "[--tensor_name=tensor_to_print]")
    sys.exit(1)

  else:
    print_tensors_in_checkpoint_file(FLAGS.file_name, FLAGS.tensor_name)
    try:
      '''
      extract the weights and biases (in hidden1) from the checkpoint file
      '''
      reader = tf.train.NewCheckpointReader(FLAGS.file_name)
      var_to_shape_map = reader.get_variable_to_shape_map()
      w_hidden1=reader.get_tensor('hidden1/weights')
      b_hidden1=reader.get_tensor('hidden1/biases')

      '''
      #############################
      load the testing data 
      #############################
      '''
      data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
      
      '''
      #############################
      Visualize the testing data
      #############################
      '''

      '''
      test=data_sets.test.images[1,:].reshape((28,28))
      rescaled = (255.0 / test.max() * (test- test.min())).astype(np.uint8)
      img = Image.fromarray(rescaled)
      img.show()
      '''


      '''
      ###############################
      Store the tf.variable in numpy
      ###############################
      '''
      init = tf.initialize_all_variables()
      sess = tf.InteractiveSession()
      sess.run(init)
      print("biases in hidden layer 1")
      print(b_hidden1)


      '''
      ##################################################################
      Visualize the weights in hidden1
      There 128 features, we first visualize the first one
      ###################################################################
      '''

      '''
      data = w_hidden1[:, 1].reshape((28,28))
      rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
      img = Image.fromarray(rescaled)
      img.show()
      '''


      '''
      ##################################################################
      Visualize the weights in hidden1
      There 128 features, we first chose to Visualize the first 64 and
      show it 8x8 
      ###################################################################
      '''

      '''
      f = pylab.figure()
      for i in range(1,64):
        data = w_hidden1[:, i].reshape((28,28))
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        img = Image.fromarray(rescaled)
        f.add_subplot(8, 8, i)  # this line outputs images on top of each other
        #pylab.imshow(img)
        pylab.imshow(img,cmap='gray')
      pylab.title('hidden1/weights')
      pylab.show()
      '''


      '''
      ####################################################################
      Use the 1st Testing image and plug into the first hidden layer:
        Feature_1= Test_1 * Weigth_1+ Biases_1
      Note: Feature_1: 1 X 128
            Test_1: 374 X 1 (a picture, which is digit: 2)
            Weigth_1: 374 X 128 
            Biases_1: 1 X 128
      Plot the Feature_1 (1 X 128)
      ######################################################################
      '''

      
      test1=np.matrix(data_sets.test.images[1,:])
      h1=np.dot(test1,w_hidden1)+np.matrix(b_hidden1)
      print(h1)
      plt.plot(np.transpose(h1))
      plt.ylabel('hidden1 features for test image 1')
      plt.figure()
      

      '''
      feature1=np.dot(test1,w_hidden1[:,1])
      f1=feature1.reshape((28,28))
      rescaled = (255.0 / f1.max() * (f1- f1.min())).astype(np.uint8)
      img = Image.fromarray(rescaled)
      img.show()
      '''
 
      '''
      ########################################
      plot the biases in hidden1 (128 X 1 )
      ########################################
      '''        
      plt.plot(b_hidden1)
      plt.ylabel('hidden1/biases')
      plt.show()

    except Exception as e:  # pylint: disable=broad-except
      print(str(e))
      if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
              "with SNAPPY.")


if __name__ == "__main__":
  tf.app.run()
