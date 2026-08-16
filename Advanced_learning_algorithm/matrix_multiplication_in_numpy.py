import numpy as np

AT = np.array([[200, 17]])

W = np.array([[1,-3,5],
              [-2, 4,-6]])

b = np.array([[-1,1,2]])

def sigmoid(z):
    return 1/(1+ np.exp(-z))

def Dense(AT, W, b, g):
    z = np.matmul(AT, W) + b
    a_out = g(z)
    return a_out

print(Dense(AT, W, b, sigmoid))