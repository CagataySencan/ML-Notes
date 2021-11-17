import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import  confusion_matrix





# Natural Language Processing (Doğal Dil İşleme)

# Herhangi bir doğal dil metninden işlem yapmaktır.
# Hedefler : NLU (Natural Language Understanding), NLG (Natural Language Generation)
# Yaklaşımlar : 
    # Lingusitik (Dilbilim) Yaklaşımı : 
        # Morphology (Şekilbilim) : Kelime ekleri ve kökleri nelerdir, farklı anlamları nelerdir ?
        # Syntax (Sözdizim) : Bir kelimenin alabileceği bütün ekleri ve anlamları alıp cümledeki konumuna göre çıkarım yapılır
        # Semantics (Anlambilim) : Bütün cümlenin anlamını çıkartmayı çıkartmak amaçlanır
        # Pragmatics (Kullanımbilim) : Cevapların tamamen istenen anlama yönelik olmasını amaçlar 
        # Bu yaklaşımda yukarıdan aşağı aşama kaydedilir
    # İstatiksel Yaklaşımlar : 
        # N-Gram : Kelimeleri harflerine göre analiz etmektir
        # TF-IDF : Sözcüğün bulunduğu dökümanı ne kadar temsil ettiğini gösteren bir istatistiksel yaklaşımdır
        # Word-Gram : Kelimelerin sayısı
        # BOW (Bag of Words) : Metin sınıflandırma

# NLP Örnekleri :
    # Yazar Tanıma (Metinden yazar çıkarımı.)
    # Metin Sınıflandırma (Çağrı merkezi,spam,iş maili vs)
    # Duygusal Kutupsallık-Fikir Madenciliği-Duygu Analizi
    # Anormal Davranış Analizi
    # Metin Özetleme
    # Soru Cevaplama
    # Etiket Bulutları-Anahtar Kelime Çıkarımı

# Kütüphaneler : 
    # NLTK : nltk.org
    # SpaCy : spacy.io
    # Stanford NLP
    # OpenNLP : Apache : opennlp.apache.org  
    # RAKE
    # Amueller World Cloud
    # TensorFlow : Word2Vec

# Türkçe Kütüphaneler : 
    # Zemberek
    # ITU
    # Tspell
    # YTU Kemik
    # Wordnet
    # TrMorph
    # TSCorpus
    # Metu

# Genel Problemler ve Yaklaşımlar
# Adımlar :
    # 1- Veri kaynağı bulunur
    # 2- Preprocess yapılır, problemler halledilir
    # 3- Öznitelik çıkarımı yapılır (N-Gram, TF-IDF)
    # 4- Öğrenim
    # 5- Sonuçlar
# Problemler

data = pd.read_csv('yorumlar.csv',error_bad_lines=False) 
    
# Space Matrix ve Imla işaretleri 
# Anlamsız ifadelere stop words denir 

 #  "[^a-zA-Z]" filtrelemedir, noktalama işaretlerinden kurtulduk

# Büyük Küçük Harf


# Stop Words (Anlamsız kelimelerin temizlenmesi)
nltk.download("stopwords")
# Kelime Gövdeleri   (stopwords olmadan)
ps = PorterStemmer() # kelime köklerine ayıracağız
derlem = []
for i in range(0,716) : 
    yorum = re.sub("[^a-zA-Z]",' ',data["Review"][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
# Öznitelik(Feature) Çıkarımı 
cv = CountVectorizer(max_features=2000)
x = cv.fit_transform(derlem).toarray()
y = data.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)
x_train = x_train.fillna(x_train.mean,inplace=True)
gnb = GaussianNB()
gnb.fit(x_train,y_train)










     