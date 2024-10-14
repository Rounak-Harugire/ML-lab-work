import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("house_prices_dataset.csv")
print("Original Data : \n",dataset)
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print("X Part : \n",x)
print("Y Part : \n",y)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print("x_train_scaled : \n",x_train_scaled)
print("x_test_scaled : \n",x_test_scaled)
from sklearn.svm import SVR
svr_model = SVR(kernel='rbf') 

 # You can also try 'linear' or 'poly'

svr_model.fit(x_train_scaled, y_train)
y_pred = svr_model.predict(x_test_scaled)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
plt.figure(figsize=(10, 6))

# Scatter plot for Actual and Predicted Prices

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Price', s=100)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Price', s=100)

# Line connecting Actual Prices

plt.plot(range(len(y_test)), y_test, color='blue', linestyle='-', marker='o', alpha=0.5)

# Line connecting Predicted Prices

plt.plot(range(len(y_pred)), y_pred, color='red', linestyle='--', marker='x', alpha=0.5)
plt.title('Actual vs Predicted House Prices', fontsize=14)
plt.xlabel('House Index', fontsize=12)
plt.ylabel('Price (in $1000)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()