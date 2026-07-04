import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=2)


np.random.seed(42)
num_samples = 100

sizes = np.random.randint(500, 3000, size=(num_samples, 1))
bedrooms = np.random.randint(1,6, size =(num_samples, 1))
floors = np.random.randint(1,4, size=(num_samples ,1))
ages = np.random.randint(0,80, size=(num_samples ,1))


X_train = np.hstack((sizes , bedrooms, floors , ages)).astype(np.float64)

y_train = (
    sizes.flatten() * 110
    - bedrooms.flatten() * 20
    - floors.flatten() * 30
    - ages.flatten() * 35
    + 360
    + np.random.normal(0,15000,num_samples)
)


X_features = ["size(sqrf)", "bedrooms","floors", "ages"]

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(
    f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}"
)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:  w: {w_norm}, b:{b_norm}")

y_pred_sgd = sgdr.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm

print(
    f"prediction using np.dot() and sgdr.predict match: {np.allclose(y_pred, y_pred_sgd)}"
)

print(f"\nPrediction on training set:\n{y_pred[:4]}")
print(f"Target values \n{y_train[:4]}")



fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label="target", alpha=0.6)
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:, i], y_pred, color="orange", label="predict")
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("target versus prediction using z-score normalized model")

plt.tight_layout()  # Tự động căn chỉnh đồ thị cho đẹp
plt.show()