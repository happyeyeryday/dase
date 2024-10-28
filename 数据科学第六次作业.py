import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
dataframe = pd.read_csv('bike.csv')
dataframe = dataframe.drop(columns=['id'])
dataframe = dataframe[dataframe['city'] == 1]
dataframe = dataframe.drop(columns=['city'])

# Modify hour values
dataframe.loc[dataframe['hour'] >= 19, 'hour'] = 0
dataframe.loc[dataframe['hour'] <= 5, 'hour'] = 0
dataframe.loc[dataframe['hour'] != 0, 'hour'] = 1

# Prepare target and features
target = dataframe['y'].values.reshape(-1, 1)
features = dataframe.drop(columns=['y']).values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train_scaled)

# Make predictions
predictions = regressor.predict(X_test_scaled)

# Evaluate the model
rmse = mean_squared_error(y_test_scaled, predictions, squared=False)
print(rmse)
