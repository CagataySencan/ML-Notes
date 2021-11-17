import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import openpyxl
import tensorflow as tns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression

# Prediction-1 Simple Linear Regression

# Ay-Satış tahmini
# Verileri yükleme
datas = pd.read_csv('satislar.csv')

# Önişleme
aylar = datas[["Aylar"]]
satislar = datas[["Satislar"]]


aylar2 = aylar.values
satislar2 = satislar.values
 
# Eğitim ve Test için ayır
x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

"""
#Verileri Ölçekle
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)
Y_train = scaler.fit_transform(y_train)
Y_test = scaler.fit_transform(y_test)
"""

# Model inşaası
lr = LinearRegression()
lr.fit(x_train,y_train)
predict = lr.predict(x_test)

# Verinin görselleştirilmesişş
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
























