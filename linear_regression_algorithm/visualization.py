import numpy as np
import matplotlib.pyplot as plt

def generate_large_dataset():
    """
    Hàm sinh dữ liệu giả lập lớn gồm 50 căn nhà để kiểm tra thuật toán.
    Chia dữ liệu thành 40 căn huấn luyện (Train) và 10 căn kiểm tra (Test).
    """
    np.random.seed(42)
    X_large = np.zeros((50, 4))
    
    X_large[:, 0] = np.random.randint(600, 3000, size=50)
    X_large[:, 1] = np.random.randint(1, 6, size=50)
    X_large[:, 2] = np.random.randint(1, 4, size=50)
    X_large[:, 3] = np.random.randint(0, 40, size=50)

    y_large = (X_large[:, 0] * 0.15 + X_large[:, 1] * 15 + X_large[:, 2] * 20 - X_large[:, 3] * 0.8 + 50 
               + np.random.normal(0, 10, size=50))
    
    X_train, X_test = X_large[:40], X_large[40:]
    y_train, y_test = y_large[:40], y_large[40:]
    
    return X_train, y_train, X_test, y_test


def plot_cost_history(J_history):
    """
    Hàm vẽ biểu đồ đường thể hiện quá trình giảm chi phí (Cost History) qua các vòng lặp.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(J_history, color='blue', linewidth=2, label="Chi phí J")
    plt.title("Lịch sử Hội tụ của Hàm Chi phí (Cost History)", fontsize=14, fontweight='bold')
    plt.xlabel("Số vòng lặp (Iterations)", fontsize=12)
    plt.ylabel("Giá trị Chi phí (Cost)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def plot_predictions_vs_targets(y_train, train_preds, y_test, test_preds):
    """
    Hàm vẽ biểu đồ phân tán (Scatter Plot) so sánh Giá thực tế và Giá dự đoán.
    Nếu các điểm nằm sát đường chéo 45 độ, mô hình dự đoán cực kỳ chính xác.
    """
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_train, train_preds, color='orange', alpha=0.7, label='Dữ liệu Huấn luyện (Train)', edgecolors='k')
    plt.scatter(y_test, test_preds, color='cyan', alpha=0.9, label='Dữ liệu Kiểm tra mới (Test)', marker='s', edgecolors='k')

    max_val = max(np.max(y_large), np.max(train_preds)) if 'y_large' in globals() else 600
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label="Đường dự đoán lý tưởng")
    
    plt.title("So sánh Giá Dự đoán vs Giá Thực tế", fontsize=14, fontweight='bold')
    plt.xlabel("Giá thực tế (Target)", fontsize=12)
    plt.ylabel("Giá dự đoán (Prediction)", fontsize=12)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
