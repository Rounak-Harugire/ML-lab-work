import numpy as np
arr = np.array([ 4, 5, 6, 7, 8, 9])
x = np.searchsorted(arr, 7)
print(x)