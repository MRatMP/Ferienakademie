# Inception network architectural graph

# Tensorflow imports
import tensorflow as tf

from HelperFunctions import models_helper

# define model parameters here
# ...

  
def inception_block(x, channels_in, filter_1, filter_3, red_filter_3, filter_5, red_filter_5, red_filter_pool, scope):
        
    with tf.variable_scope(scope):
        # to implement 
        # ...
    return y
    
def forward(images_batch, training):

    with tf.name_scope('inceptionnet'):
        
        # to implement
        # ...
        
    return output
        
    
