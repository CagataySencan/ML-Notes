import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import theano 


# Deep Learning

# Artificial Neural Network
# Göreceli olarak yavaş çalışırlar ve fazla kaynak isterler
# Girdiler sinapsisler üzerinden taşınır, nöronlar karar verip çıktı yollar
# Nöron : Karar veren sistemdir 
# Bağımsız değişkenler, standart olarak scale edilmiş olmalıdır (0-1 aralığında olmalı)
# Bağımlı değişken, binomial nominal kategorik değerler olabilir
# Çoktu çıktı elde edilebilir
# Sinapsisler üzerinde ağırlıklar taşınır, ağırlıklar sinyali etkileyerek çıktıyı etkiler
# Nöronu aksiyona sokan fonksiyona aktivasyon fonksiyon denir 
# Ne kadar fazla öznitelik verirseniz verin ANN içinde yapılabilir
# Elle yapılan preprocess kısımlaru ANN içinde yapılır

# Aktivasyon Fonksiyonları 

# Threshold (Eşik) Fonksiyonları
# Belli bir eşik değerinden sonra sinyal verip 1 değeri döndürür. Sinyal döndürmezse 0 olarak değer döner.
# Sigmoid Fonksiyonu (Adım Fonkiyonu), Rectifier (Düzleştirimiş) Fonksiyon
# Hyperbolic Tangent Fonksiyon

# Katman (Layer) Kavramı

# Input-Hidden-Output katmanları bulunur
# Girdiler OR ve AND kapısı mantığıyla çalışır
# XOR gate çözümü için nöron sayısı artırmak gerekir 
# OR ve AND kapıları tek bir çizgiyle ayrılabiliyorken XOR kapısı tek bir çizgiyle ayrılamaz
# Linearly seperable değildir 

# ANN öğrenme yöntemi ve Perceptron (Algılayıcı) Kavramı

# ANN gerçek değer ile tahmin değerini karşılaştırıp aradaki farkın çok olması durumunda
# w1 ve w2 ağırlıkları üzerinde değişiklikler yaparak bu farkı kapatmaya çalışır
# Geri besleme değeri c = 1/2(gerçek-tahmin)^2 bu değere göre bir learning rate belirlenerek
# ağırlıklar yeniden düzenlenir. learning rate git gide düşürülen bir çarpandır

# Stochocastic Gradien Descendent (Gradiyent Alçalış)

# Gradient Descendent bizi geri dönüş yaparken izlememiz gereken optimum noktaları
# bulmamıza yardımcı olur. Bunu bulan bir algoritmadır.

# Algoritmada her veri sonunda learning rate'i alçaltma veya değiştirmeme kararı alınır 

# Batch Gradient Descendent 
# Bu algoritmada her veri okunduktan sonra learning rate hakkında karar verilir
# Mini Batch'de ise veri gruplarına bakarak karar verilir 

# ANN Öğrenme Algoritması

# Adımlar : 
    # 1- Ağdaki threshold ve ağırlık değerleri 0-1 aralığındaki rastgele sayılarla ilklendir
    # 2- Veri kümesinden ilk satır giriş katmanından verilir (her öznitelik bir nöron)
    # 3- İleri yönlü yayılım yapılarak ANN istenen sonucu verene kadar güncellenir
    # 4- Gerçekle çıktı arasındaki fark alınarak hata hesaplanır 
    # 5- Geri yayılım yapılarak her sinapsis üzerindeki ağırlık, hataya olan etkisine ve
    # öğrenme oranına bakılarak güncellenir.
    # 6- İstenen sonuç elde edilene kadar 1-5 arasındaki adımlar tekrarlanır
    # 7- Bütün eğitim kümesi çalıştırıldıktan sonra bir tur (epoch) tamamlanmış olur
    # aynı veri kümesi kullanılarak tekrar tekrar tur yapılır 
    
    
# Deep Learning Kütüphaneleri : 
    # Caffe : Berkeley'de geliştirilmiş bir kütüphanedir
    # PyTorch : Yeni yükselen bir kütüphane 
    # TensorFlow 
    # Keras 
    
    
    
# Müşteri Kayıp Analizi Uygulaması 

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

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# ANN Oluşturma 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6,activation="relu",input_dim=11))
classifier.add(Dense(6,activation="relu"))
classifier.add(Dense(6,activation="relu"))
classifier.add(Dense(1,activation="sigmoid"))
classifier.compile(optimizer='adam',loss="binary_crossentropy" ,metrics= ["accuracy"])

classifier.fit(X_train,y_train,epochs=50)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
y_pred = (y_pred>0.5)

cm = confusion_matrix(y_test,y_pred)




















