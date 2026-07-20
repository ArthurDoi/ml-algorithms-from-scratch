import numpy as np
import pandas as pd

np.random.seed(42)

exam1 = np.random.uniform(30, 100, 100)
exam2 = np.random.uniform(30, 100, 100)

noise = np.random.normal(0, 5, 100)
admitted = ((exam1 + exam2 + noise) >= 130).astype(int)

data = pd.DataFrame({
    'Exam_1_score': np.round(exam1, 4),
    'Exam_2_score': np.round(exam2, 4),
    'Admitted': admitted
})

data.to_csv('ex2data1.csv', index=False)
print("Đã tạo xong file ex2data1.csv thành công!")
