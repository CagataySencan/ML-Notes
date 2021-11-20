import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dimension Reduction
# Kullanım Alanları :
    # Gürültü Filtreleme
    # Görselleştirme
    # Öznitelik Çıkarımı
    # Öznitelik Eleme/Değiştirme
    # Borsa Analizi 
    # Sağlık Verileri/Genetik Veriler

# Birincil Bileşen Analizi (Principal Component Analysis)
# Boyut indirgemek bazen veri kaybına yol açabilir
# Adımlar :
    # İndirgenmek isteyen boyut k olsun
    # Veri Standartlaştırılır
    # Covariance ve Corelliation Matrisinden özdeğer ve özvektörleri elde et (ya da SVD il)
    # Öz değerleri büyükten küçüğe sırala ve k tanesini al 
    # Seçilen k tane özdeğerin W projeksiyon matrisini oluştur
    # Orjinal veri kümesi X'i W kullanarak dönüştür ve K-boyutlu Y uzayını elde et

# PCA Kodlaması : 

veriler = pd.read_csv("Wine.csv")

X = veriler.iloc[:,0:13].values
Y = veriler.iloc[:,13].values

from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=42)
classifier.fit(X_train,y_train) # pca öncesi

classifier2 = LogisticRegression(random_state=44)
classifier2.fit(X_train2,y_train) # pca sonrası

y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)


from  sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm2 = confusion_matrix(y_test,y_pred2)
cm3 = confusion_matrix(y_pred,y_pred2)

# LDA (Linear Discriminant Analysis)
# PCA'den farklı olarak sınıflar arası ayrımı gözetir ve maksimize eder
# PCA unsupervised LDA supervised özelliklidir 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_trainLDA =  lda.fit_transform(X_train,y_train)
X_testLDA = lda.transform(X_test)

classifierLDA = LogisticRegression(random_state=0)
classifierLDA.fit(X_trainLDA,y_train)

y_predLDA = classifierLDA.predict(X_test),
cm4 = confusion_matrix(y_test,y_predLDA)










