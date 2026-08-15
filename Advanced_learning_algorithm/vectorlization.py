import numpy as np

W = np.array([[3],[2]])

X_batch = np.array([[[4,5],[1,2],[3,0],[2,6]]])

result = np.matmul(X_batch, W)

print(result)
