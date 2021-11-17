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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB

# Naive Bayes
# Bayes Teoremi : Koşullu Olasılık P(Y/X) = P(X/Y)P(Y)/P(X)
# Dengesiz veri setlerinde de çalışır
# Sürekli değerlerde GaussianNB
# Multinomial değerlerde MultinomialNB
# 1 veya 0 gibi, iki durumu ayırt etmek için BernoulliNB kullanılır



datas = pd.read_csv("veriler.csv")

x = datas.iloc[:,1:4].values
y = datas.iloc[:,4:].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

gnb = GaussianNB()
gnb.fit(X_train,y_train)
yPred = gnb.predict(X_test)
cm = confusion_matrix(y_test,yPred)










