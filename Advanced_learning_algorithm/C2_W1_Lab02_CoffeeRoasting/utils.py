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