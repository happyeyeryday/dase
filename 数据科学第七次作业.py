import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

# 读取数据
data = pd.read_csv('fraudulent.csv')

# 查看缺失值情况
print(data.isnull().sum())

# 处理缺失值
# 可以选择剔除缺失值过多的列或使用众数填充等方法
# 这里选择使用众数填充
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 分离特征和标签
X = data_imputed.drop(columns=['y'])
y = data_imputed['y']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 创建并训练逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算F1值
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.4f}')
