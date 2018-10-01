
import os.path

import tensorflow as tf

from Models import DenseNet as model
from Data import input_data
from HelperFunctions import train_helper
from HelperFunctions import models_helper

# parameter
LEARNING_RATE = 0.00001
MAX_STEPS = 800 # steps = train_steps/batch_size
MAX_VALID_STEPS = 200 # steps = valid_steps/valid_batch_size
MAX_TEST_STEPS = 200  # steps = test_steps/test_batch_size
MAX_EPOCHS = 15
TRAIN_STEPS = 40000 # = how many training images 
TEST_STEPS = 10000 # = how many test images
VALID_STEPS = 10000 # = how many validation images
BATCH_SIZE = 50
BATCH_SIZE_TEST = 50
BATCH_SIZE_VALID = 50
OPTIMIZER = 2 # 1 = SGD, 2 = Adam, 3 = RMSProp


def run_training():
    
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    LOG_DIR = os.path.join(BASE_DIR, 'logs/')
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'trained_models/')
    
    mean, std = input_data.get_mean_std()

    with tf.Graph().as_default():
      
        # to implement
        # ...
        # define graph and its operations here

        for epoch in range(MAX_EPOCHS):

            print('EPOCH : ', epoch, ' BEGIN')
            
            # to implement
            # ...
            
            for step in range(MAX_STEPS):
                
                # to implement
                # ...

                # Save a checkpoint of the model after every epoch.
                if (step + 1) == MAX_STEPS:
                    print('Saving current model state')
                    # to implement
                    # ...

            # to implement
            # ...
            
            print('EPOCH : ', epoch, ' END')

        print('\n-----------------------------------------------')
        print('Run trained model on test data: ')
        # to implement
        # ...
            
if __name__ == '__main__':
    run_training()
