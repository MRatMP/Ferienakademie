# Define overall training function, like e.g. data reading or validation during training.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from Data import input_data 


def placeholder_inputs(batch_size, height_size, width_size, channels_size, output_size):
    
    # to implement 
    # ...

    return images_placeholder, labels_placeholder


def do_eval(sess, accuracy, images_placeholder, labels_placeholder, batchnorm_placeholder, is_valid, steps, batch_size):
   
    # to implement
    # ...
    
    return mean_error
