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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
# Desicion Tree
# Sürekli bölme işlemleri yapılarak datalar sınıflandırılır
# Belli bölgeye düşen datalara göre ayrım yapılır ve tahmin ona göre yapılır

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
plt.show()

# Tahmin
print(linReg2.predict(polyReg.fit_transform([[14]])))

sc1 = StandardScaler()
xScaled = sc1.fit_transform(X)
sc2 = StandardScaler()
yScaled = sc2.fit_transform(Y)
# Support Vector Regression

# İlk kullanımı sınıflandırma içindir
# Bir marjin aralığında maksimum noktayı almak için kullanılır. Marjin dışı noktalar hata olarak adlandırılır
# Marjin değerini minimize eden doğru çizilir
# Scaler ile kullanılmak zorundadır 

svrRegression = SVR(kernel="rbf")
svrRegression.fit(xScaled,yScaled)

plt.scatter(xScaled,yScaled)
plt.plot(xScaled,svrRegression.predict(xScaled))
plt.show()

desicionTreeRegressor = DecisionTreeRegressor(random_state=0)
desicionTreeRegressor.fit(X, Y)

plt.scatter(X,Y)
plt.plot(X,desicionTreeRegressor.predict(X))
plt.show()

print(desicionTreeRegressor.predict([[10]]))
 

# Ensemble Learning (Kolektif Öğrenme)
# Random Forest Algoritması
# Veriyi birden çok karar ağacına böler
# Her ağaç farklı durumları ele alır
# Bu farklı ağaçlar bir ağaca alır ve çoğunluğun dediği olarak ana karar ağacı kurar

rfr = RandomForestRegressor(n_estimators=10,random_state=0) # kaç ağaç çizileceği 

rfr.fit(X,Y.ravel())
print(rfr.predict([[10]]))
























