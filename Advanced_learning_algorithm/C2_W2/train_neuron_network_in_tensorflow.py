from tabnanny import verbose

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy


X = np.array([[0.5, 1.5], [1.0, 2.0], [1.5, 0.5], [2.0, 1.0]], dtype=np.float32)
Y = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)

model = Sequential([
    Dense(units = 25, activation ='sigmoid'),
    Dense(units = 15, activation ='sigmoid'),
    Dense(units = 1, activation ='sigmoid'),
])

model.compile(loss=BinaryCrossentropy())

model.fit(X,Y,epochs = 1000, verbose=0)

test_preds = model.predict(X)
print(test_preds)