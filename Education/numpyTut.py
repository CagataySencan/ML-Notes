import numpy as np
## Numpy Arrays
myList = [10,20,30]
myList2 = [30,20,10]
myList3 = [20,30,10]
matrix = np.array([myList,myList2,myList3])
print(matrix) ## matris tanımlama


## Numpy Methods
print(np.arange(0,15)) ## range ile aynı
print(np.arange(0,15,2))
print(np.zeros(3))
print(np.zeros((3,3))) ## sıfır matrisi
## ones diye aynı method sadece 1 leri içerir
print(np.linspace(15,30,4)) ## bitiş indeksi dahil
print(np.identity(4)) ## birim matris
print(np.random.randn(4)) ## np array şeklinde random sayılar döndürür
print(np.random.randn(4,4)) ## matris formu
## randint random ile aynı falan size verilmeli


##Array methodları
exampleList = np.random.randint(0,100,25)
print(exampleList)
print(exampleList.reshape(5,5)) ## arrayı matrise döndürür
print(exampleList.max()) ## maxı bulur
print(exampleList.min())
print((exampleList.argmax())) ## maxsimum ve minimumun yerini bulur (indeksi değil)
print(exampleList.argmin())
print(exampleList.reshape(5,5).shape) ## matris boyutunu döndürür

## Numpy indexes
## indeksler liste ile aynı
print(exampleList[3:5]) ## liste ile aynı bitiş dahil değil slice işlemi
# ## listeyi bölüp bölünen listeyi değiştirirsen ana liste de değişir
exampleList[3:9] = 5 ## aralıktakileri verilene eşitler
print(exampleList)
##exampleList[:] = 31 ##listedeki bütün elemanları sayıya eşitler
print(exampleList)
## eğer alt liste ana listeyi etkilemesin diyorsan exampleListCopy = exampleList.copy() yapmalısın

##Matrix indexes
##indeksler satırlara denk gelir
## matris indeksleri matlab ile aynı
print(matrix[0:,2]) ## 0. rowdan sonuncu rowa kadar sadece 2. elemanları al
print(matrix[0,:2])

#Numpy Operations
newList = exampleList < 31
print(exampleList[newList]) ## yeni listeye atama yaparak ve bu şekilde çağırarak ana listedeki istediğin elemanları bulursun
## iki dizi arasındaki matematiksel işlemler yapılırkan birebir endekslenerek yapılır ve aynı matlabdaki gibi bir kare alma kök alma vs durumunda bütün verilerin tek tek işlemi yapılır
