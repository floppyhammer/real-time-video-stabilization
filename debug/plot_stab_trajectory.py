import numpy as np
from numpy import polynomial as P
import matplotlib as mpl
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 3, figsize=(12, 4))

data = np.loadtxt("trajectory.txt")
smoothed_data = np.loadtxt("smoothed_trajectory.txt")

print(data)
print(print(data))

x = data[:, 0]

ax[0].plot(x, data[:, 1], label='x')
ax[0].plot(x, smoothed_data[:, 1], label='x2')

ax[1].plot(x, data[:, 2], label='y')
ax[1].plot(x, smoothed_data[:, 2], label='y2')

ax[2].plot(x, data[:, 3], label='a')
ax[2].plot(x, smoothed_data[:, 3], label='a2')

for a in ax:
    a.legend(loc='lower right', fontsize=15)

plt.show()
