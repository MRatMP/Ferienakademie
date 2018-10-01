import numpy as np

def get_input_output_pairs():
    sinogram = np.ones((256, 256), dtype=np.float32)
    image = np.ones((256, 256), dtype=np.float32)
    return sinogram, image