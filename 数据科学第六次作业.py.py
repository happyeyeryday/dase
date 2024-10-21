import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np  # 用于计算平方根

# 1. 读取 CSV 文件
file_path = 'bike.csv'  # 确保此文件与您的脚本在同一目录中
bike_data = pd.read_csv(file_path)

# 2. 数据预处理：删除 'id' 列
bike_data = bike_data.drop(columns=['id'])

# 3. 筛选出上海市的数据
shanghai_data = bike_data[bike_data['city'] == 1]  # 1 代表上海

# 4. 删除 'city' 列
shanghai_data = shanghai_data.drop(columns=['city'])

# 5. 统一 'hour' 列的值
shanghai_data['hour'] = shanghai_data['hour'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

# 6. 提取 'y' 列并转换为 NumPy 列向量
y = shanghai_data['y'].to_numpy().reshape(-1, 1)  # 转换为列向量
shanghai_data = shanghai_data.drop(columns=['y'])  # 删除 'y' 列

# 7. 将 DataFrame 转换为 NumPy 数组
X = shanghai_data.to_numpy()  # 将处理后的 DataFrame 转换为 NumPy 数组

# 8. 数据集划分：按照 8:2 的比例划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. 数据预处理 VI：归一化处理
scaler_X = MinMaxScaler()  # 创建 MinMaxScaler 实例
scaler_y = MinMaxScaler()  # 创建 MinMaxScaler 实例，用于 y 的归一化

# 归一化训练集和测试集的特征数据
X_train_scaled = scaler_X.fit_transform(X_train)  # 归一化训练集特征
X_test_scaled = scaler_X.transform(X_test)        # 归一化测试集特征

# 归一化训练集和测试集的标签
y_train_scaled = scaler_y.fit_transform(y_train)  # 归一化训练集标签
y_test_scaled = scaler_y.transform(y_test)        # 归一化测试集标签

# 10. 模型构建：构建线性回归模型
model = LinearRegression()  # 创建线性回归模型实例

# 11. 利用训练集训练模型
model.fit(X_train_scaled, y_train_scaled)  # 训练模型

# 12. 利用测试集进行预测
y_pred_scaled = model.predict(X_test_scaled)  # 预测测试集标签

# 13. 反归一化预测值和测试集标签
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_inverse = scaler_y.inverse_transform(y_test_scaled)

# 14. 计算均方误差 (MSE)
mse = mean_squared_error(y_test_inverse, y_pred)

# 15. 计算均方根误差 (RMSE)
rmse = np.sqrt(mse)  # 计算 RMSE

# 16. 输出 RMSE 值
print("均方根误差 (RMSE):", rmse)
