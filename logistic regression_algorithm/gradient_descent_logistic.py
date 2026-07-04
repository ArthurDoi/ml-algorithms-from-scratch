from sigmoid import sigmoid
import numpy as np
import copy, math
import numpy as np
from cost import compute_cost_logistic
import matplotlib.pyplot as plt


# X_train = np.array([[0.5,1.5],[1,1],[1.5, 0.5],[3, 0.5],[2,2],[1, 2.5]])
# y_train = np.array([0,0,0,1,1,1])

def compute_gradient_logistic(X,y,w,b):
    """
        Computes the gradient for logistic regression

        Args:
          X (ndarray (m,n): Data, m examples with n features
          y (ndarray (m,)): target values
          w (ndarray (n,)): model parameters
          b (scalar)      : model parameter
        Returns
          dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
          dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """

    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)  #(n,)(n,)=scalar
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_db, dj_dw

def gradient_descent(X,y,w_in, b_in,alpha, num_iters):
    """
       Performs batch gradient descent

       Args:
         X (ndarray (m,n)   : Data, m examples with n features
         y (ndarray (m,))   : target values
         w_in (ndarray (n,)): Initial values of model parameters
         b_in (scalar)      : Initial values of model parameter
         alpha (float)      : Learning rate
         num_iters (scalar) : number of iterations to run gradient descent

       Returns:
         w  (ndarray (n,))   : Updated values of parameters
         b (scalar)        : Updated value of parameter
    """
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X,y,w,b)
        w = w - alpha *dj_dw
        b = b - alpha *dj_db
        if i < 100000:
            J_history.append(compute_cost_logistic(X,y,w,b))
        if i % math.ceil(num_iters / 10 ) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w,b, J_history

X_train = np.array([[0.],[1],[2],[3],[4],[5]])
y_train = np.array([0,0,0,1,1,1])

w_tmp = np.zeros_like(X_train[0])
b_tmp = 0
alph = 0.1
iters = 10000
w_out, b_out, J_history = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

plt.figure(figsize=(8,5))

plt.plot(J_history, color='blue', linewidth = 2, label="cost function")
plt.title("quá trình hội tụ Gradient Descent", fontsize=14, fontweight='bold')
plt.xlabel("số vòng lặp iteration", fontsize=12)
plt.ylabel("giá trị chi phí",fontsize=12)
plt.grid(True, linestyle='--', alpha = 0.6)
plt.legend()
plt.show()
# X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
# y_tmp = np.array([0, 0, 0, 1, 1, 1])
# w_tmp = np.array([2.,3.])
# b_tmp = 1.
# dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
# print(f"dj_db: {dj_db_tmp}" )
# print(f"dj_dw: {dj_dw_tmp.tolist()}" )




