import numpy as np
import copy, math
def compute_cost(X,y,w,b):
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) **2
    cost = cost / (2*m)
    return cost

def compute_gradient(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j]  = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw /m
    dj_db = dj_db /m

    return dj_db, dj_dw

def gradient_descent(X,y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(X,y,w,b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"loop {i:4d}: cost {J_history[-1]:8.2f}")
    return w ,b, J_history
    
