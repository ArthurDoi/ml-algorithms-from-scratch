import tensorflow as tf
import numpy as np

layer_1 = tf.keras.layers.Dense(units=1, activation='sigmoid')
x = np.array([[1.0,2.0,3.0]])
a1 = layer_1(x)
a2 = a1.numpy()
print(a1)
print(a2)