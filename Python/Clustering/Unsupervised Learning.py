import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch



# Clustering
# Müşteri segmentasyonu : geçmiş verilere göre öneri verme, ortak davranışlara bakılarak pazarlama 
# Tehdit ve sahtekarlık yakalama 
# Özel kampanya önerileri
# Segmentasyona göre eksik veri tamamlama 
# Pazar segmentasyonu : Davranışlar, Demografik, Psikolojik, Coğrafi segmentasyonlarla gruplar oluşturulur.
# Bu segmentasyonlarla pazar analizi yapılabilir.
# Sağlık ve Görüntü işleme

# Unsupervised Learning (Gözetimsiz Öğrenme)
# Gözetimsiz öğrenmede eğitim verisi yoktur. Algoritma grupları öğrenmeye çalışarak yeni verileri
# en uygun gruba atar.

# K-Means (K-Ortalama)
# 5 Adımla çalışır 
# 1-Kaç küme olacağını gireriz 
# 2-Rasgele k noktası seçilir
# 3-Her yeni veri örneği en yakın merkez noktasına göre atanır
# 4-Her küme için yeni merkez noktası hesaplanılarak yeni merkez noktaları seçilir
# 5- Tekrarlanır

# Rassal Başlangıç Noktası Tuzağı
# Bölütler içerisindeki noktaların birbirine uzaklığı minimum
# Bölütlerin birbirine olan uzaklığı maksimum olmalıdır
# Bunların sağlanmaması durumunda başlangıç noktalarımız yanlış seçilmiş olur
# Bunu çözmek için farklı çözümler vardır 
# K-Means++ algoritması seçilen merkez noktaların diğer bütün noktalara olan uzaklığı bulunur D(x) 
# D(x)^2 ile yeni noktaların olasılığı bulunur
# (Araştır)

# Bölüt sayısına karar verme 
# WCSS hesabı yapılır (within-clusters sum of squares) 
# WCSS değerleri cluster artırılarak grafiğe dökülür, bu grafikte bir dirsek noktasında optimum bulunur
# (Eğimin değişmemeye başladığı nokta)

# K-Means kodu


datas = pd.read_csv("musteriler.csv")

x = datas.iloc[:,3:]


kmeans = KMeans(n_clusters = 3,init ="k-means++")
kmeans.fit(x)
print(kmeans.cluster_centers_)
sonuclar = []
# for i in range (1,11) : 
#     kmeans = KMeans(n_clusters = i,init ="k-means++",random_state = 123)
#     kmeans.fit(x)
#     sonuclar.append(kmeans.inertia_)
yPred1 = kmeans.fit_predict(x)
k_labels = np.unique(yPred1)
for i in k_labels:
    plt.scatter(x.iloc[yPred1 == i,0],x.iloc[yPred1 == i,1])
plt.legend()
plt.show()

# Hiyerarşik Bölütleme 
# Agglomerative Adımları
# 1-Her veri tek bir bölütle başlar
# 2-En yakın iki komşu alınarak yeni bölütler oluşturulur
# 3-En yakın iki kümeyle yeni bölütler oluşturulur
# 4-Önceki adım tek bölüt kalana kadar devam eder

# Divisive Adımları ise Agglomerative Adımlarının tam tersidir
# Cluster arasındaki mesafeler ölçülürken farklı referanslar alınabilir
# En yakın iki komşu, en uzak iki komşu, ortalama, merkezler arası mesafeler

# Dendogram 
# Clusterların nasıl birleştiğini ve aradaki mesafesini gösteren grafiksel yapıdır
# Bu mesafeleri bulmak için mesafe matrisi yapılır

ac = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
yPred2 = ac.fit_predict(x)
# Görselleştirme 
h_labels = np.unique(yPred2)
for i in h_labels:
    plt.scatter(x.iloc[yPred2 == i,0],x.iloc[yPred2 == i,1])
plt.legend()
plt.show()

ac = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
dendrogram = sch.dendrogram(sch.linkage(x,method="ward"))












