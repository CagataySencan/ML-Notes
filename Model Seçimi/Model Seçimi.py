# Model Seçimi 
# Probleme göre model seçimi

# K-Fold Validation 
# Aldığımız k değeri kadar veri bölünür 
# Bu bölünmelerle verinin tamamı hem eğitim hem test için kullanılmış olur 
# Her aşamada accuracy alınıp ortalama başarı bulunur 
# basari = cross_val_score(classifier,Xtrain,Ytrain,cv = katman sayısı)

# Model tipine göre algoritma belirleme 
# Bağımlı Değişken var mı ?
# Bağımlı değişken kategorik mi sürekli sayı mı ?
# Linear mi nonLinear mi ?

# Grid Search 
# Grid Search elde olan parametrelerle uyguladığımız algoritmaları optimize etmeye yarar
# Kodlaması için scikitlearn'den dökümantasyona bakılabilir 
