import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import tensorflow as tf
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import  Dense

## Tensorflow
"""
Nöron (Perceptron)
buradaki oluşturduğumuz nöronlar da gerçek nöronlar gibi çalışır
birden fazla girdi alıp çıktı verir 
bias denilen sabitler vardır
katmanlı bir yapıyla deep network kurulur
girdi katmanları gizli katmanlar ve çıktı katmanı

Aktivasyon fonksiyonları 

Sigmoid Fonksiyonu
0-1 arası değer alır. genelde sınıflandırmada kullanılır

Tanh(hiperbolik tanjant) Fonksiyonu 
-1-1 arası değer alır  negatif değerlerle daha geniş kapsam sağlar 
sınıflandırma problemlerinde kullanılır.

ReLU (Rectified Linear Unit)
0 ile sonsuz arasında değer alır 
derin öğrenme alanında kullanılır

Linear fonksiyonlar 
sonsuz değer alabilir fakat non-linear olmaması modellerde sorun olur

Tensorflow Giriş
seaborn kütüphanesi detaylı ve güzel gözüken grafikler için kullanılır
sbn.pairplot() 3 farklı grafik çizer

"""
medalsFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")
# y = wx + b aranan şey label
y = medalsFrame["Fiyat"].values
# x -> özellikler
x = medalsFrame[["BisikletOzellik1","BisikletOzellik2"]].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)
# dataları uygun hale getirir
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#nöronlar oluşturulur
model = Sequential()
model.add(Dense(3,activation="relu")) # kaç tane eklersen  o kadar hidden layer olr
model.add(Dense(3,activation="relu"))
model.add(Dense(3,activation="relu"))
model.add(Dense(3,activation="relu"))
model.add(Dense(1)) #output node eklenir
model.compile(optimizer="rmsprop",loss="mse")
model.fit(x_train,y_train,epochs=250)
trainLoss = model.evaluate(x_train,y_train,verbose=0)
testLoss = model.evaluate(x_test,y_test,verbose=0)
testTahmin = model.predict(x_test) # dataya göre tahmin
tahminFrame = pd.DataFrame(y_test,columns= ["Gerçek Y"])
testTahmin = pd.Series(testTahmin.reshape(330,))
tahminFrame = pd.concat([tahminFrame,testTahmin],axis=1)
tahminFrame.columns = ["Gerçek Fiyat","Tahmin Fiyat"]
model.save("xxxx.h5") # kaydetme

# dataframe.isnull().sum() hangi rowda kaç null veri var gösterir
# dataframe.corr()["kolondaki data"] ## kolondaki datanın diğer datalar tarafından nasıl etkilendiğini gösterir
# çok uç veriler modeli patlatabileceği için hem çıkararak hem çıkarmayarak model denenir
