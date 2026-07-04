import numpy as np
from gradient_function import gradient_descent, compute_cost, compute_gradient
from z_score_normalization import zscore_normalize_features

from visualization import generate_large_dataset, plot_cost_history, plot_predictions_vs_targets

X_train, y_train, X_test, y_test = generate_large_dataset()

X_train_norm, mu, sigma = zscore_normalize_features(X_train)

initial_w = np.zeros((X_train_norm.shape[1],))
print(initial_w)
initial_b = 0.0

iterations = 1500
alpha = 0.1

print("Training 40 House")
w_final, b_final, J_hist = gradient_descent(
    X_train_norm, y_train, initial_w, initial_b, 
    compute_cost, compute_gradient, alpha, iterations
)

train_preds = np.zeros(y_train.shape)
for i in range(X_train_norm.shape[0]):
    train_preds[i] = np.dot(X_train_norm[i], w_final) + b_final

X_test_norm = (X_test - mu) / sigma
test_preds = np.zeros(y_test.shape)
for i in range(X_test_norm.shape[0]):
    test_preds[i] = np.dot(X_test_norm[i], w_final) + b_final

print("\nTest test")
for i in range(len(y_test)):
    print(f"Nhà mới {i+1}: Dự đoán: {test_preds[i]:0.2f}k$, Thực tế: {y_test[i]:0.2f}k$ | Lệch: {abs(test_preds[i]-y_test[i]):0.2f}k$")

plot_cost_history(J_hist)
plot_predictions_vs_targets(y_train, train_preds, y_test, test_preds)
