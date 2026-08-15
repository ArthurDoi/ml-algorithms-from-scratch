import numpy as np


W = np.array([[3],[2]])

X_batch = np.array([ [[4,5], [1,2]],[[3,0],[2,6]]])
ket_qua = np.matmul(X_batch, W)
unit =  X_batch.shape[1]
ket_qua1 = X_batch @ W
print(ket_qua1)
print(ket_qua)
print(unit)