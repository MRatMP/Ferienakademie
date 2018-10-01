import tensorflow as tf


def batch_norm(x, training, scope):
    
    # to implement
    # ...
    
    
    return x

def conv2d(x, channels_in, channels_out, kernel_size, stride_size, padding_mode, use_bias, scope):
    
    # to implement
    # ...
        
    return y

def loss(predictions, gt_labels):
    
    # to implement
    # ...
    
    return cross_entropy
    
def training(loss, learning_rate, optimizer_num):
    """
    Sets up the training Ops.

    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

    Returns:
    train_op: The Op for training.
    """
    # to implement
    # ...
   
    return train_op
