import numpy as np
import matplotlib.pyplot as plt

x = (np.arange(100)-50)/50

print(x[0])

plt.plot(x, np.abs(x+0.5))
plt.show()