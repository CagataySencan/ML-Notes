import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
import math





# Reinforced Learning (Pekiştirmeli/Takviyeli Öğrenme)
# Robotlar ve Oyunlarda kullanılır
# Makine eylemlerinden çıkarım yaparak eylemlerini iyileştirir
# Geri besleme sistemi gibidir
# Tek Kollu Canavar : 
# A/B testi : Örneğin iki farklı reklamın kullanıcı beğenisine sunup feedbacke göre bir reklam seçimi A/B testidir.



# Rasgele Yaklaşım


# N = 10000
# toplam = 0
# secilenler = []
# for n in range(0,N) :
#     ad = rnd.randrange(10)
#     secilenler.append(ad)
#     odul = datas.values[n,ad]
#     toplam = toplam + odul
    
# plt.hist(secilenler)    
# Upper Confidence Bound(Üst Güven Sınırı)
# Her olayın arkasında bir dağılım olduğunu savunur
# Amaç dağılımları maksimuma çıkarmaktır
# Adım 1 : Her turda, i'nin tıklanma sayısı ve o reklamdan gelen ödül tutulur
# Adım 2 : Ortalama ödül ve güven aralığı hesaplanır 
# Adım 3 : En yüksek UCB değerine olan seçenek alınır
# N = 10000
# oduller = [0] * 10
# tiklamalar = [0] * 10 
# toplam2 = 0 
# secilenler = []
# for n in range(0,N) :
#     maxUCB = 0
    
#     ad = 0
#     for i in range(0,10):
#         if tiklamalar[i] > 0 : 
#             ortalama = oduller[i] / tiklamalar[i]
#             delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i])
#             UCB = ortalama + delta
#         else:
#              UCB = N*10
#         if maxUCB < UCB :
#             maxUCB = UCB
#             ad = i
#     secilenler.append(ad)
#     tiklamalar[ad] = tiklamalar[ad] + 1
#     odul = datas.values[n,ad]
#     oduller[ad] = oduller[ad] + odul
  
#     toplam2 = toplam2 + odul
            


# Thompson Örneklemesi
# Eldeki datanın ne kadar doğru olduğunu bulmaya yarar
# Elimizdeki küçük bir data parçasından büyük datanın dağılımı tahmin edilir
# Adım 1 : Her aksiyon için gelen (Ni1(n)) ve gelmeyen (Ni0(n)) ödül sayısı hesaplanır 
# Adım 2 : Beta değeri hesaplanır 
# En yüksek beta değerine sahip seçenek kullanılır




datas = pd.read_csv("reklamlar.csv")


N = 10000
d = 10
oduller = [0] * d
tiklamalar = [0] * d 
toplam3 = 0 
secilenler = []
birler = [0] * d
sifirlar = [0] * d

for n  in range(0,N):
    ad = 0
    maxTh = 0
    for i in range (0,d):
        rasBeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        if(rasBeta > maxTh):
            maxTh = rasBeta
            ad = i
    secilenler.append(ad)
    odul = datas.values[n,ad]
    if (odul == 1):
        birler[ad] = birler[ad] + 1
    else : 
        sifirlar[ad] = sifirlar[ad] + 1 
    toplam3 = toplam3 + odul





















