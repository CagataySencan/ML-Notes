import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpydoc
import openpyxl as op

## Matplotlib Library

yalnizList = np.array([1,2,2,3,4,4,4,5,7,8])
uzgunlukList = np.array([1,2,3,4,4,5,6,6,9,11])
plt.plot(yalnizList,uzgunlukList,"y")
plt.xlabel("Yalnız gün sayısı")
plt.ylabel("Üzgünlük durumu")
plt.title("Yalnızlığın üzgünlüğe etkisi")
##plt.show()
#grafik çizdirme (çizgi grafiği)

list3 = np.linspace(0,15,10)
lisDepreshun = list3 **3
plt.subplot(1,2,1) ## bir sıra 2 grafik birincisi
plt.plot(list3,lisDepreshun,"g") ## çizgi
plt.subplot(1,2,2) # bir sıra 2 grafik ikincisi
plt.plot(lisDepreshun,list3,"r")
##yanyana grafik çizdirme
##plt.plot(list3,lisDepreshun,"g--") ##kesikli çizgi
##plt.plot(list3,lisDepreshun,"g*-") ##yıldızlı çizgi
plt.xlabel("Yalnız gün sayısı")
plt.ylabel("Üzgünlük durumu")
plt.title("Yalnızlığın üzgünlüğe etkisi")

myFigure = plt.figure(figsize=(5,5)) ## figür boyutu ayarlanabilir
myFigureAxes = myFigure.add_axes([0.3,0.3,0.6,0.6]) ## 0.6'lar boy ve en 0.3'ler ise iç içe grafik çizdirdiğinde grafiklerin nerede olacağı
myFigureAxes.plot(list3,lisDepreshun,"r*-")
myFigureAxes.set_xlabel("Günlük Çikolata Sayısı")
myFigureAxes.set_ylabel("Mutluluk artışı")
myFigureAxes.set_title("Çikolatanın Mutluluğa Etkisi")

##myFigure.savefig("myFigure.png",dpi = 500) ## figürü dpi(kalite) vererek kaydetme
## plot objesi oluşturarak grafik çizdirme
## iç içe grafikler
##i = int(input("Please enter columns : "))
##j = int(input("Please enter rows : "))
(myFigure2,myAxes) = plt.subplots(figsize = (10,10),nrows=2,ncols=2) # bu ilk terimi figür ikincisi eksen olan bir tuple döndürür
## burada da fisgsize verilerek figür boyutu ayarlanabilir
for axes in myAxes :
    if myAxes.shape >= (2,2) :
        for i in axes :
            i.plot(list3,lisDepreshun,label="Depresyon artışı")
            i.legend()
            i.set_xlabel("Yalnız Gün Sayısı")
            i.set_ylabel("Depresyon etkisi")
            i.set_title("Yalnızlığın Depresyona etkisi")
    else :
        axes.plot(list3, lisDepreshun,label="Depresyon Eğrisi") ## label eğrinin ismidir
        axes.legend() ## eğrinin ismini yazdırır buna lokasyon vererek grafikteki yerini ayarlayabilirsin
        axes.set_xlabel("Yalnız Gün Sayısı")
        axes.set_ylabel("Depresyon etkisi")
        axes.set_title("Yalnızlığın Depresyona etkisi")
plt.tight_layout()

## aynı anda birden fazla grafik döndürme (row*col kadar grafik döndürür, ve burada eksen sayısı (nparray) fazla olduğundan 2 loop kullanılır).

#Görsel geliştirmeler##
## aynı eksende birdemn fazla eğri çizdirilebilir
## bunu eksen.plot şeklinde yapabilirsin
arrayOne = np.linspace(0,50,5)
arrayTwo = arrayOne**2
(myNewFigure,myNewAxes) = plt.subplots()
##myNewAxes = plt.plot(arrayOne,arrayTwo,"#8FBDFF",linewidth=2, linestyle = ":",marker = "o", markersize = 10,markerfacecolor = "r")
##myNewAxes = plt.plot(arrayTwo,arrayOne,"r")
## grafiklerde eğrileri belirli renklerde çizmek yerine rgb hex kodu yazılabilir
##linewidth kalınlık ayarlar

## grafik çeşitleri
#plt.scatter(arrayOne,arrayTwo) #nokta grafik
#plt.hist(arrayTwo) histogram 
#plt.boxplot(arrayTwo) boxplot






