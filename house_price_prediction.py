import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset with an absolute path
data = pd.read_csv('d:/vscode/intern/Hunarpe/divesh/house price data.csv')

# Data preprocessing
data = data.dropna()
data = data.drop_duplicates()

# Convert date columns to numeric or drop them if they are not relevant
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].astype('int64') // 10**9  # Convert to Unix timestamp

# Ensure all features are numeric
X = data.select_dtypes(include=[np.number])
y = data['price']

# Drop the target variable from features
X = X.drop(columns=['price'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the training set
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Training MSE: {train_mse}')
print(f'Training R2 Score: {train_r2}')

# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test MSE: {test_mse}')
print(f'Test R2 Score: {test_r2}')

# Plotting the actual vs predicted prices
plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
