from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import theano 


# XGBoost (Extreme Gradient Boosting)
# Yüksek verilerde iyi performans gösterir
# Hızlı çalışır
# Modelin yorumlanması kolaydır

veriler = pd.read_csv("Churn_Modelling.csv")

X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X[:,1] = le.fit_transform(X[:,1])
X[:,2] = le.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohn = ColumnTransformer([("ohn",OneHotEncoder(dtype=float),[1])],remainder="passthrough")

X = ohn.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

classifier = XGBClassifier()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()

# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)

# # ANN Oluşturma 

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# classifier = Sequential()
# classifier.add(Dense(6,activation="relu",input_dim=11))
# classifier.add(Dense(6,activation="relu"))
# classifier.add(Dense(6,activation="relu"))
# classifier.add(Dense(1,activation="sigmoid"))
# classifier.compile(optimizer='adam',loss="binary_crossentropy" ,metrics= ["accuracy"])

# classifier.fit(X_train,y_train,epochs=50)
# y_pred = classifier.predict(X_test)

# from sklearn.metrics import confusion_matrix
# y_pred = (y_pred>0.5)

# cm = confusion_matrix(y_test,y_pred)