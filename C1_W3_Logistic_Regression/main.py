from cost_function import compute_cost
import copy
import math
from utils import *
import numpy as np
from gradient_descent import gradient_descent, compute_gradient
file_path = "./ex2data1.csv"
X_train, y_train = load_data(file_path)

# print("First five elements in X_train are:\n", X_train[:5])
# print("Type of X_train:",type(X_train))

# print("First five elements in y_train are:\n", y_train[:5])
# print("Type of y_train:",type(y_train))

# print ('The shape of X_train is: ' + str(X_train.shape))
# print ('The shape of y_train is: ' + str(y_train.shape))
# print ('We have m = %d training examples' % (len(y_train)))

m,n = X_train.shape

# initial_w = np.zeros(n)
# initial_b = 0
# cost = compute_cost(X_train, y_train, initial_w, initial_b)
# print('Cost at initial w and b (zeros): {:.3f}'.format(cost))

# initial_w = np.zeros(n)
# initial_b = 0

# dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
# print(f'dj_db at initial w and b (zeros):{dj_db}' )
# print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)

# plot_decision_boundary(w, b, X_train, y_train)


p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))