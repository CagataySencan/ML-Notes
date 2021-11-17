import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import openpyxl
import tensorflow as tns
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Polynomial Regression

datas = pd.read_csv("maaslar.csv")

x = datas.iloc[:,1:2]
y = datas.iloc[:,2:]
X = x.values
Y = y.values

linReg= LinearRegression()
linReg.fit(X,Y)


plt.scatter(X,Y)
plt.plot(x,linReg.predict(X))
plt.show()

# Polynomial Model
polyReg = PolynomialFeatures(degree=2)
xPoly = polyReg.fit_transform(X)

linReg2 = LinearRegression()
linReg2.fit(xPoly,y)
plt.scatter(X,Y)
plt.plot(X,linReg2.predict(polyReg.fit_transform(X)))

# Tahmin
print(linReg2.predict(polyReg.fit_transform([[14]])))