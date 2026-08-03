import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def sigmoid(z):
    
    g = 1 / (1 + np.exp(-z))
    return g


def linear_neuron():
    X_train = np.array([[1.0], [2.0]], dtype=np.float32)
    Y_train = np.array([[300.0], [500.0]], dtype=np.float32)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
    ax.legend(fontsize='large')
    ax.set_ylabel('Price (in 1000s of dollars)')
    ax.set_xlabel('Size (1000 sqft)')
    plt.show()

    linear_layer = tf.keras.layers.Dense(units=1, activation='linear')
    print("Weights trước build:", linear_layer.get_weights())

    a1 = linear_layer(X_train[0].reshape(1, 1))
    print("Output a1:", a1.numpy())

    w, b = linear_layer.get_weights()
    print("w:", w, "b:", b)

    set_w = np.array([[200]])
    set_b = np.array([100])
    linear_layer.set_weights([set_w, set_b])
    print("Weights sau set:", linear_layer.get_weights())

    a1 = linear_layer(X_train[0].reshape(1, 1))
    alin = np.dot(set_w, X_train[0].reshape(1, 1)) + set_b
    print("Layer output:", a1.numpy())
    print("Numpy output:", alin)


def activation_logistic_neuron():
    X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)
    Y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)
    X_new = np.array([[1.0], [1.5], [2.0], [2.5], [3.0]], dtype=np.float32)
    pos = Y_train == 1
    neg = Y_train == 0

    model = Sequential([
        Dense(1, input_dim=1, activation="sigmoid", name="L1")
    ])
    model.summary()

    logistic_layer = model.get_layer('L1')
    print("Weights trước set:", logistic_layer.get_weights())

    set_w = np.array([[2]])
    set_b = np.array([-4.5])
    logistic_layer.set_weights([set_w, set_b])
    print("Weights sau set:", logistic_layer.get_weights())

    a1 = model.predict(X_train[0].reshape(1, 1), verbose=0)
    
    predictions = model.predict(X_new, verbose=0)
    print(predictions)
    alog = sigmoid(np.dot(set_w, X_train[0].reshape(1, 1)) + set_b)
    print("Model predict:", a1)
    print("Numpy sigmoid:", alog)

    x_range = np.linspace(-1, 6, 100).reshape(-1, 1)
    y_curve = sigmoid(set_w[0, 0] * x_range + set_b[0])

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(x_range, y_curve, label='sigmoid(w*x+b)', color='blue')
    ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c='red', label="y=1")
    ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0",
               facecolors='none', edgecolors='blue')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('x')
    ax.set_ylabel('probability')
    ax.legend()
    plt.show()


if __name__ == "__main__":   
    linear_neuron()
    activation_logistic_neuron()