import numpy as np
import matplotlib.pyplot as plt
def load_coffee_data():
    """
    12-15 minutes is best
    temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2) # khởi tạo bộ sinh số ngẫu nhiên cố định trên mọi máy
    X = rng.random(400).reshape(-1,2)
    X[:,1] = X[:,1] * 4 + 11.5
    X[:,0] = X[:,0] * (285-150) + 150

    Y = np.zeros(len(X))
    i = 0
    for t, d in X:
        y = -3/(260-175) * t + 21
        if(t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1
    return X, Y.reshape(-1,1)



def plot_binary_data (X,Y):
    plt.figure(figsize=(8,6)) # tạo khung bản vẽ
    y_flat = Y.flatten()
    X_good = X[y_flat == 1]
    plt.scatter(X_good[:, 0], X_good[:, 1],
                marker='x', color='red', s=80,label='Good Roast')

    X_bad = X[y_flat ==0]
    plt.scatter(X_bad[:, 0], X_bad[:, 1], 
                marker='o', facecolors='none', edgecolors='royalblue', s=80, label='Bad Roast')

    t_line = np.linspace(175, 260,100)
    y_line = -3 / (260-175) * t_line + 21
    plt.plot(t_line, y_line, color='purple',  linestyle='-', linewidth=1.5)
    plt.axvline(x=175, color='purple', linestyle='-', linewidth=1)
    plt.axhline(y=12.0, color='purple', linestyle='-', linewidth=1)
    
    # 4. Trang trí đồ thị
    plt.title("Coffee Roasting - Data Visualization", fontsize=14, fontweight='bold')
    plt.xlabel("Temperature (Celsius)", fontsize=11)
    plt.ylabel("Duration (minutes)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right') # Hiển thị bảng chú thích

    plt.show()


def sigmoid(z):
    return 1 / (1+ np.exp(-z))


def plt_network(X, Y, netf):
    x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 0.1))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds = netf(grid_points)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    y_flat = Y.flatten()
    
    Z1 = preds.reshape(xx.shape)
    contour1 = ax1.contourf(xx, yy, Z1, levels=20, cmap='Blues', alpha=0.3)
    fig.colorbar(contour1, ax=ax1) # Thanh đo xác suất từ 0.0 đến 1.0
    
    ax1.scatter(X[y_flat==1, 0], X[y_flat==1, 1], marker='x', color='red', label='Good Roast')
    ax1.scatter(X[y_flat==0, 0], X[y_flat==0, 1], marker='o', facecolors='none', edgecolors='royalblue', label='Bad Roast')
    ax1.set_title("Network Probability")
    ax1.set_xlabel("Temperature"); ax1.set_ylabel("Duration")
    ax1.legend()

    Z2 = (preds >= 0.5).astype(int).reshape(xx.shape)
    ax2.contourf(xx, yy, Z2, levels=[0.5, 1.0], colors=['orange'], alpha=0.2)
    ax2.contour(xx, yy, Z2, levels=[0.5], colors=['purple'], linewidths=1.5)
    
    ax2.scatter(X[y_flat==1, 0], X[y_flat==1, 1], marker='x', color='red', label='Good Roast')
    ax2.scatter(X[y_flat==0, 0], X[y_flat==0, 1], marker='o', facecolors='none', edgecolors='royalblue', label='Bad Roast')
    ax2.set_title("Network Decision")
    ax2.set_xlabel("Temperature"); ax2.set_ylabel("Duration")
    ax2.legend()

    plt.tight_layout()
    plt.show()
