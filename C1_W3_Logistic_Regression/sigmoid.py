import numpy as np

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


# val = -1
# print(sigmoid(val))