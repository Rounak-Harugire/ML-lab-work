import pandas as pd
import numpy as np

df=pd.read_csv("Data.csv")
print('Original Data : \n',df)

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imp.fit(x[:,1:3])
x[:,1:3]=imp.transform(x[:,1:3])
print('transform Data : \n',x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
T = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
P = np.array(T.fit_transform(x))
print('For Non-Numerical-Country- data For X Part :\n',P)

from sklearn.preprocessing import LabelEncoder
L = LabelEncoder()
y = L.fit_transform(y)
print('For Y Part : \n',y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print("X Part of Train :",x_train)
print("Y Part of Train :",y_train)

# Normalization Or Min_Max Scale
# X_min = (X - X_min) / (X_min - X_max)

from sklearn.preprocessing import MinMaxScaler
nm = MinMaxScaler()
# x_train[:, 3:] = nm.fit_transform(x_train[:, 3:])
# x_test[:, 3:] = nm.transform(x_test[:, 3:])

x_train[:, 1:3] = nm.fit_transform(x_train[:, 1:3])  # Adjust based on your actual data columns
x_test[:, 1:3] = nm.transform(x_test[:, 1:3])

print("Normalised X Part of Train :\n",x_train[:, 1:3])
print("Normalised X Part of TEST :\n",x_test[:, 1:3])