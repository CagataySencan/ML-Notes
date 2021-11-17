import pandas as pd
import numpy as np
## Pandas Library

## Series
myDickt = {"Çağatay" : "Sporcu","Taha" : "Bilinmiyor","Barış" : "Sporcu"}
print(pd.Series(myDickt)) ## Exceldeki table a benzer, tanımlanması böyledir
myData = ["Sporcu","Bilinmiyor","Sporcu"]
print(pd.Series(myData))  ## otomatik indeksler
myTitle =["Çağatay","Taha","Barış"]

print(pd.Series(myData,myTitle)) ## birebir indeksleyerek tablo(seri) oluşturur
print(pd.Series(index=myTitle,data=myData)) ## aynısı
## liste dict dışında numpy arrayi ve tuple  olarak da girdi verilebilir

##Series özellikleri
## datalara ulaşmak için direkt index yazılır
mySeries = pd.Series(myData,myTitle)
mySeries2 = pd.Series(myData,myTitle)
mySeries3 = mySeries + mySeries2 ## seriler böyle toplandığında aynı indexteki elemanlar toplanır
print(mySeries["Taha"]) ## örnek
print(mySeries3) ## örnek

## Data Frame
randomMatrix = np.random.randint(100,size=(4,4))
dataFrame = pd.DataFrame(randomMatrix,index= ["1","2","3","4"],columns=["1","2","3","4"])
print(dataFrame)
## bu data frame a normal liste olarak inderksleme denenirse indeksteki column gelir
## indeksleme işleminde aynı anda birden fazla column okunabilir
print(dataFrame[["1","2"]])
print(dataFrame.loc["1"]) ## rowa ulaşmak için bu yapı kullanılır

##Data Frame indeksleme
dataFrame["5"] = [1,2,3,4] # column ekleme
print(dataFrame)
dataFrame.loc["5"] = [1,2,3,4,5] # row ekleme
print(dataFrame)
dataFrame.drop("5",axis=1,inplace=True) ## column silme
dataFrame.drop("5",inplace=True) ## row silme
print(dataFrame)
## kalıcı olarak silmek istenmiyorsa inplace false yapılmalı ya da hiç yazılmamalı
print(dataFrame.loc["1"]["1"])
print(dataFrame.loc["1","1"])
## aynı işlem, belirli dataya ulaşmak için yapılır.
print(dataFrame.iloc[1]) ## row isminden bağımsız olarak normal indeksleme gibi row getirir
print(dataFrame > 0)
booleanFrame = dataFrame > 0
print(dataFrame[booleanFrame]) ## aynı listelerdeki gibi aranan sayıları bulmak için
print(dataFrame[dataFrame["1"] > 0]) ##bu işlemde belirli columdaki istediğin sayıdan büyük ve küçük sayıları bulabilirsin,olmayanları elimine edersin

## İndeks değiştirme
yeniIndeksler = ["Çağatay","Taha","Barış","Mete"]
dataFrame["Yeni İndeksler"] = yeniIndeksler
dataFrame.set_index("Yeni İndeksler",inplace=True)
print(dataFrame)
##  colum indekslerini değiştirmek için tabloya yeni bir column eklenir ardından o column yukardaki yöntemle yeni indeks olur

##Multi İndeksler
firstIndexes = ["Sporcu","Sporcu","Bilinmiyor","Sporcu Değil"]
innerIndexes = ["Çağatay","Barış","Taha","Mete"]
mergedIndexes = list(zip(firstIndexes,innerIndexes))
mergedIndexes = pd.MultiIndex.from_tuples(mergedIndexes)
matrix = np.array([[21,"Tekirdag"],[22,"Giresun"],[24,"Malatya"],[22,"Çorum"]])
multiDataFrame = pd.DataFrame(matrix,index=mergedIndexes,columns=["Yaş","Memleket"])
print(multiDataFrame)
## bu yöntem belli bir gruptaki özellikleri multi indexleyerek grup haline getirir, aynı özellikteki datalar için ayrı ayrı yazım olmaz
print(multiDataFrame.loc["Sporcu"])
print(multiDataFrame.loc["Bilinmiyor"].loc["Taha"]) ## erişim
print(multiDataFrame[["Yaş"]])
multiDataFrame.index.names = ["Özellik","İsim"]
print(multiDataFrame)

## Eksik veriler
## data frame de  .dropna ile olamyan verilerin olduğu rowları çıkartabilirsin axis eklersen kolonu getirir
## thresh = ile sayı verirsen o sayı ve üstündeki nan verilerin olduğu yerleri yok eder
## .fillna metodu verdiğin değeri nan değerlere atar

## Gruplandırma
#groupby methodu verilen veriye göre rowları gruplar
print(multiDataFrame.groupby("Yaş").count())
#print(multiDataFrame.groupby("Yaş").mean())
print(multiDataFrame.groupby("Yaş").max())
print(multiDataFrame.groupby("Yaş").describe())

##Concat
#birden fazlı aynı patternde gelen dataları birleştirmektir
## pd.concat[frame1,frame2,frame3....]

#Merge
##pd.merge(frame1,frame2,on = "Ortak column")

##İleri Pandas operasyonları
## frame["Column].unique --> sadece özel dataları getirir, tekrarlayan datalardan sadace bir tane gelir
print(dataFrame["1"].unique())
##.nunique bunun sayısını verir
print(dataFrame["1"].nunique())
##.value_counts() verilen columndaki değerlerden kaçar tane olduğunu gösterir
print(dataFrame["1"].value_counts())
## .apply(fonskiyon) ile seçilen columndaki bütün elemanları fonksiyona yollayıp işlem yapar
def carp(num) :
    return num**2
print(dataFrame["1"].apply(carp))
##null kontrolü .isnull()
print(dataFrame["1"].isnull())
## pivot ile multiindex

innerIndexes1 = [["Sporcu","Çağatay",21,"Tekirdag"],["Sporcu","Çağatay",32,"Tekirdag"],["Sporcu","Barış",22,"Giresun"],["Bilinmiyor","Tuha",24,"Malatya"],["Sporcu Değil","Mete",22,"Çorum"]]
pivotFrame = pd.DataFrame(innerIndexes1,columns=["Spor Durumu","İsim","Yaş","Memleket"])
print(pivotFrame)
print(pivotFrame.pivot_table(values=["Yaş"],index=["Spor Durumu","İsim","Memleket"],aggfunc=np.mean))
##pivot_table sonuna aggfunc = fonksiyon ekleyerek
##excel dosyasını python projesiyle aynı yere kaydederek, .read_excel("isim.xlsx") ile excel dosyasını okuyabilirsin
##.to_excel("isim.xlsx") ile dataframe i excel olarak kaydedebilirsin
pivotFrame.to_excel("pivotframe.xlsx") ## excele yazdırma
print(pd.read_excel("pivotframe.xlsx")) ## excelden okuma