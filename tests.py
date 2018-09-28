import numpy as np

coordinates = np.array([10,10])
dims = np.array([5,5])

print(np.all(np.greater(dims, coordinates)))
