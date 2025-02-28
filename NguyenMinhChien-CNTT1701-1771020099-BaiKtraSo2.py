import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Đọc file Excel
df = pd.read_excel("Ket_qua_hoc_tap_RMIT.xlsx")

# In dữ liệu gốc
print("Dữ liệu gốc:")
print(df)

# Phát hiện giá trị thiếu
df_cleaned = df.copy()
missing_values = df_cleaned.isnull().sum()
print("\nSố lượng giá trị thiếu trong mỗi cột:")
print(missing_values)

# Xử lý giá trị thiếu (thay bằng trung bình của cột tương ứng)
df_cleaned.fillna(df_cleaned.mean(numeric_only=True), inplace=True)

# In dữ liệu sau khi làm sạch
print("\nDữ liệu sau khi làm sạch:")
print(df_cleaned)

# Tính toán các chỉ số thống kê cơ bản
descriptive_stats = df_cleaned.describe()
print("\nCác chỉ số thống kê cơ bản:")
print(descriptive_stats)

# Vẽ biểu đồ cột cho điểm tổng kết
df_cleaned["Điểm tổng kết"].plot(kind='bar', title='Biểu đồ cột - Điểm tổng kết', color='skyblue')
plt.xlabel("Sinh viên")
plt.ylabel("Điểm tổng kết")
plt.show()

# Vẽ biểu đồ phân phối
df_cleaned["Điểm tổng kết"].plot(kind='hist', bins=10, title='Biểu đồ phân phối - Điểm tổng kết', color='orange', edgecolor='black')
plt.xlabel("Điểm tổng kết")
plt.ylabel("Tần suất")
plt.show()

# Chuẩn bị dữ liệu
X = df_cleaned[["Điểm giữa kỳ", "Điểm cuối kỳ"]]
y = df_cleaned["Điểm tổng kết"]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nĐánh giá mô hình:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R^2 Score: {r2}")