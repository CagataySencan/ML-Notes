import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import openpyxl
import tensorflow as tns
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression

#   Multiple Linear Regression

#   Dummy Variable
#   Asıl data yerine kullanılan kullanılan datalara denir.Örneğin asıl datasetteki şehir ismi yerine plaka kullanmak.
#   Aynı anda bu iki data kullanılmamalıdır.
#   Birden fazla Dummy Variable varsa en uygun olan seçilmelidir.

#   P-Value(Olasılık Değeri) : Genelde 0.05 alınır.
#   H0 : null hypothesis : Boş hipotez
#   H1 : Alternatif Hipotez 
#   P değeri küçüldükçe H0, büyüdükçe H1'in hatalı olma olasılığı artar.

#   Değiken Seçimi

#   Değişkenlerin bağımlı değişkene etkisini ölçmek ve bağımsız değişkenleri seçmek amacıyla farklı yaklaşımlar vardır
#   1-Bütün değişkenleri dahil etmek
#   2-Geriye Doğru Eleme (Backward Elemination)  -
#   3-İleri Seçim (Forward Selection)             -> Adım Adım Karşılaştırma (Stepwise)
#   4-İki Yönlü Eleme (bidirectional Elemination)-
#   5-Skor Karşılaştırması (Score Comparison)

#   1- Bazı durumlarda kullanılır, bütün değişkenler dahil edilir
#   2- Bütün değişkenler dahil edilir. Başarı değeri tanımlanır. Model kurulur. En büyük P-Value sahibi
#   değişken tesbit edilir P>SL ise o değişken kaldırılır ve model güncellenir. Tersi durum olana kadar 
#   bu adımlarla devam edilir.


#   İleriye doğru seçim
#   1- SL seçilir(Genelde 0.05)
#   2-Bütün değişkenler içerilen bir model inşaa edilir 
#   3-En düşük P-Value olan değişken ele alınır 
#   4-Bu aşamada 3. adımdaki değişken sabit tutularak bi değişken daha eklenir 
#   5- Model güncellenir ve 3. adıma dönülür. P<SL sağlanıyorsa 3. adıma dönülür, aksi takdirde biter.

#   Çift Yönlü Eleme 
#   1- SL seçilir(Genelde 0.05)
#   2- Bütün değişkenler kullanılarak bir model inşaa edilir
#   3- En düşük P değerine sahip değişken ele alınır
#   4- Bu aşamada 3. aşamada seçilen değişken sabit tutularak bütün değişkenler modele dahil edilir
#   ve en düşük p değerine sahip olan değişken sistemde kalır 
#   5- SL değerinin altında olan değişkenler sistemde kalır ve eski değişkenlerden hiçbiri sistemden ayrılmaz
#   6- Öğrenim sonlanır 

#   Genel Yöntem
#   1- Başarı kriteri belirlenir
#   2- Bütün olası regresyon yöntemeleri inşaa edilir
#   3- Kriteri en iyi sağlayan yöntem seçilir 
#   4- Öğrenim sonlanır 

#   Multiple Linear Regression için veri hazırlama 
 
datas = pd.read_csv("veriler.csv")

# Nümerik verilere dönüşüm için encoding 
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
ulke = datas.iloc[:,0:1].values
ulke[:,0] = le.fit_transform(datas.iloc[:,0])
ulke = ohe.fit_transform(ulke).toarray()

cinsiyet = datas.iloc[:,-1:].values
cinsiyet[:,0] = le.fit_transform(datas.iloc[:,-1:])
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
cinsiyet = cinsiyet[:,0:1]


yas = datas.iloc[:,1:4].values


# Dizileri DataFrame'e Döndürmek

cinsiyet1 = pd.DataFrame(data = cinsiyet,index=range(22),columns=["Cinsiyet"])
ulke1 = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
yas1 = pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])

frame1 = pd.concat([ulke1,yas1],axis=1)
finalFrame = pd.concat([frame1,cinsiyet1],axis=1)

# Eğitim ve Test için verileri ayırmak 

x_train,x_test,y_train,y_test = train_test_split(finalFrame,cinsiyet1,test_size=0.33,random_state=42)

# Ölçekleme 
# sc = StandardScaler()

# X_train = sc.fit_transform(x_train)
# y_train = sc.fit_transform(y_train)

# Model Oluşturmna 
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
 
# Backward Elemination

X = np.append(arr = np.ones((22,1)).astype(int),values= finalFrame,axis=1)
X_l = finalFrame.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model2 = sm.OLS(cinsiyet,X_l).fit()
print(model2.summary())

# En yüksek p değeri x6 da
X_l = finalFrame.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model2 = sm.OLS(cinsiyet,X_l).fit()
print(model2.summary()) 

# En yüksek değer x3 de 

X_l = finalFrame.iloc[:,[0,1,3,4]].values
X_l = np.array(X_l,dtype=float)
model2 = sm.OLS(cinsiyet,X_l).fit()
print(model2.summary())





















































