import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori



# Association Rule Mining (Birliktelik Kural Çıkarımı)
# Kavramın çıkış noktası tekrarlayan eylemlerdir. (Örneğin alışveriş sepetleri)
# Nedensellik öğrenmede pek önemli değildir. Korleasyon bilgisyarlar için daha önemlidir.
# Veri dersinde öğretildiği gibi 
# Support(A) =  A Varlığı İçeren Eylemler / Toplam Eylem Sayısı
# Confidence(A -> B) = A ve B Varlığını İçeren Eylemler / A Varlığını İçeren Eylemler
# Lift(Etki)(A->B) = Confidence(A->B) / Support(B)
# APriori Kullanım alanları :
# Complex Event Processing 
# Kampanya
# Davranış Tahmini
# Yönlendirilmiş ARM
# Zaman Serisi Analizi
            
# Apriori Kodu    
datas = pd.read_csv("sepet.csv",header=None)
t = []
for i in range(0,7501):
    t.append([str(datas.values[i,j]) for j in range(0,20)])
    
transaction = apriori(t,min_support = 0.01,min_confidance = 0.2,min_lift = 3,min_length = 3)
print(list(transaction))

# Eclat Algoritması (Equivaleance Class Transformation)
# Apriori Breadth (Yayılım) First Search 
# Eclat Depth First Search ile çalışır
# Daha hızlı çalışır
# Frekans yapmadan direkt  transactionlara bakılır