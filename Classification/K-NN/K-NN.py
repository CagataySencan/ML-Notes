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
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# K Nearest Neighborhood
# K değeri yeni giren dataya en yakın olan komşuları ifade eder
# Yakın komşulardan baskın olanlara göre yeni data sınıflandırılır
# K tek sayı olursa yakın komşulara olan mesafeler bakılır

datas = pd.read_csv("veriler.csv")

x = datas.iloc[:,1:4].values
y = datas.iloc[:,4:].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(X_train,y_train)

yPred = knn.predict(X_test)
cm = confusion_matrix(y_test,yPred)
print(cm)