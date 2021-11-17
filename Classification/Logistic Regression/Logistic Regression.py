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
# Sınıflandırma Algoritmaları
# Sayısal olmayan verilerin tahminine denebilir


# Logistic Regression
# Sigmoid Fonksiyonu ve Step Function kullanılır
# Sayısıal değerler üzerinden istenen sonucun tahminini yapar (Kadın-Erkek gibi)

datas = pd.read_csv("veriler.csv")

x = datas.iloc[:,1:4].values
y = datas.iloc[:,4:].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

logReg = LogisticRegression(random_state=42)

logReg.fit(X_train,y_train)

yPred = logReg.predict(X_test)

# Karmaşıklık Matrisi (Confusion Matrix)
# Gerçekle tahmin arasında ilişki kurmamıza yardım eder

# true positive false negative
# false positive true negative 
# matrisin köşegenleri üzeründeki tahminler daima tutmuştur
'''
sensitivity = tpos/(tpos + fneg) true positive recognition rate
specificity = tneg/(tneg + fpos) ture negative recognition rate
precision = tpos/(tpos + fpos)
accuary = sensitivity*pos(pos+neg) + specificity*neg(pos+neg)
recall =
algoritma doğruluğunu ölçmek için bazı yöntemler

'''
confusionMatrix = confusion_matrix(y_test,yPred)
print(confusionMatrix)













