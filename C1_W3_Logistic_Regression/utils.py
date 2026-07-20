import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sigmoid import sigmoid


def plot_decision_boundary(w, b , X, y):

    positive = X[y == 1]
    negative = X[y == 0]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(positive[:, 0], positive[:, 1], c='black', marker='+', label='y=1', s=50)
    plt.scatter(negative[:, 0], negative[:, 1], c='yellow', marker='o', edgecolors='orange', label='y=0', s=40)
    
    plot_x = np.array([30, 100])
    
    plot_y = (-1 / w[1]) * (w[0] * plot_x + b)
    
    plt.plot(plot_x, plot_y, c="b", linewidth=2)
    plt.ylabel('Exam 2 score') 
    plt.xlabel('Exam 1 score') 
    plt.xlim(25, 105)
    plt.ylim(15, 105)
    plt.legend(loc="upper right")
    plt.grid(False)
    
    plt.show()

def draw_plot(file_path):

    df = pd.read_csv(file_path)

    positive = df[df['Admitted'] == 1]
    negative = df[df['Admitted'] == 0]

    plt.figure(figsize=(8, 6))
    plt.scatter(positive['Exam_1_score'], positive['Exam_2_score'], c='black', marker='+', label='Admitted', s=50)
    plt.scatter(negative['Exam_1_score'], negative['Exam_2_score'], c='yellow', marker='o', edgecolors='orange', label='Not admitted', s=40)

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.xlim(30, 100)
    plt.ylim(30, 100)
    plt.legend(loc='upper right')
    plt.title('Scatter plot of training data (Generated)')
    plt.grid(False)

    plt.show()

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, 0:2].to_numpy()

    y = df.iloc[:, 2].to_numpy()
    return X, y


def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """

    m, n = X.shape
    p = np.zeros(m)
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += X[i,j] * w[j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        if f_wb > 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p