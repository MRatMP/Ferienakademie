# for unpacking data
import pickle
import numpy as np
import os.path

# constant values
ITEMS_PER_SET = 10000
IMAGE_SIZE = 3072
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
NUM_CLASSES = 10


def unpack_data():
    
    PATH = os.path.abspath(os.path.dirname(__file__))
    BASE_PATH = os.path.join(PATH, 'cifar-10-batches-py/')
    
    dicts = list()
    
    for i in range(1, 7):
        if i < 6:
            file_name = os.path.join(BASE_PATH, 'data_batch_' + str(i))
        else:
            file_name = os.path.join(BASE_PATH, 'test_batch')
                
        with open(file_name, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            dicts.append(data_dict)
   
    return dicts


def encode_label(label):
    
    labels_vector = np.zeros([NUM_CLASSES])
    
    # to implement 
    # ...
    
    return labels_vector


def get_mean_std():
    
    dicts = unpack_data()

    global_mean = np.zeros(IMAGE_CHANNELS)
    global_std = np.zeros(IMAGE_CHANNELS)
        
    # to implement 
    # ... 
    
    
    return global_mean, global_std


# Use first four dicts for training (40k); the fifth for validation (10k);
# The test batch for testing (10k)     
def next_train_batch(batch_size):
    
    # random choosing
    dicts = unpack_data()

    np.random.randint(4)

    current_dataset = dicts[4]

    current_images = current_dataset[b'data']
    current_labels = current_dataset[b'labels']
    current_images = np.reshape(current_images, (ITEMS_PER_SET, IMAGE_SIZE))

    indices = np.random.choice(current_images.shape[0], batch_size)

    images = current_images[indices, ...]
    labels = [encode_label(current_labels[i]) for i in indices]

    images = np.reshape(images, (-1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
    images = np.transpose(images, (0, 2, 3, 1))

    labels = np.reshape(labels, (batch_size, NUM_CLASSES))
    
    return images, labels


def next_valid_batch(batch_n, batch_size):
    
    # naive method - without randomizing and without queue processing
    dicts = unpack_data()
    
    images = np.empty([batch_size, IMAGE_SIZE])
    labels = np.empty([batch_size, NUM_CLASSES])
    
    # get the validation dataset
    current_dataset = dicts[4]
    
    current_start = batch_size*batch_n
    
    current_images = current_dataset[b'data']
    current_labels = current_dataset[b'labels']

    for i, j in enumerate(range(current_start, min(current_start + batch_size, current_images.shape[0]))):
        images[i] = current_images[j]
        labels[i] = encode_label(current_labels[j])

    images = np.reshape(images, (-1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
    images = np.transpose(images, (0, 2, 3, 1))
    labels = np.reshape(labels, (batch_size, NUM_CLASSES))
        
    return images, labels
    
    
def next_test_batch(batch_n, batch_size):
    
    # naive method - without randomizing and without queue processing
    dicts = unpack_data()
    
    images = np.empty([batch_size, IMAGE_SIZE])
    labels = np.empty([batch_size, NUM_CLASSES])
    
    # get the test dataset
    current_dataset = dicts[5]
    
    current_start = batch_size*batch_n
    
    current_images = current_dataset[b'data']
    current_labels = current_dataset[b'labels']
    
    for i, j in enumerate(range(current_start, min(current_start + batch_size, current_images.shape[0]))):
        images[i] = current_images[j]
        labels[i] = encode_label(current_labels[j])

    images = np.reshape(images, (-1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
    images = np.transpose(images, (0, 2, 3, 1))
    labels = np.reshape(labels, (batch_size, NUM_CLASSES))
        
    return images, labels
