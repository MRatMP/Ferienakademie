# DenseNet architectural graph

# Tensorflow imports
import tensorflow as tf

from HelperFunctions import models_helper

# define model parameters here
# ...
 
def dense_layer(x, channels, training, scope):
   
    with tf.variable_scope(scope):  
       # to implement
       # ...
    return x, channels
    
def dense_block(x, channels, training, scope):
    
  
    # to implement
    # ...
    return y

def dense_transition(x, training, scope):
  
    
    with tf.variable_scope(scope):
        # to implement
        # ...
    return y
          
def forward(images_batch, training):

    with tf.name_scope('densenet'):
       # to implement
       # ...
    return output
        
    
