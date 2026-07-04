import numpy as np
from gradient_function import gradient_descent, compute_cost, compute_gradient
from z_score_normalization import zscore_normalize_features
import matplotlib.pyplot as plt


x = np.arange(0,20,1)
y = 1 + x ** 2

X = np.c_[x, x**2, x**3]

print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization  
X, mu, sigma = zscore_normalize_features(X)
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")
w_init = np.zeros(X.shape[1])
b_init = 0.0

model_w, model_b, J_hist = gradient_descent(X,y, w_init, b_init,
                                             compute_cost, compute_gradient, alpha = 1e-1, num_iters = 100000)
print(f"w,b found by gradient descent: w: {model_w}, b: {model_b}")
plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("No feature engineering (Fixed)")

plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value")  
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()