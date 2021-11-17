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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from  sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Classification Bitiş
# Confusion Matrix
# Köşegen değerleri önemlidir

# False Positive-False Negative
# Accuracy Paradox
# ZeroR : Hangi sınıflandırma label'ında en çok örnek varsa onu sonuç olarak ata
# Bu yüzden sadece accuracye göre değerlendirme yanlış olur. ZeroR algoritması herhangi bir öğrenme yapmaz

# ROC Eğrisi (Recivier Operating Characteristic)
# TPR (True Positive Rate) = TP / TP + FN
# FPR (False Positive Rate) = FP / TN + FP
# Y eksenine  TPR, X eksenine FPR konularak algoritmalar buna göre işaretlenir
# Bu noktaları doğrusal olarak birleştirildiğinde algoritmalar sınıflandırılıp karşılaştırılabilir
# Bu eğriden sonra alan içinde kalan algoritmalar direkt olarak elenebilir
# Pozitif Negatif oranına göre algoritmalar sınıflandırılabilir 
# Bu üç değerlendirme sonucunda en iyi algoritma bulunabilir

# AUC Değeri ROC eğrisindeki algoritmalara ne kadar hızlı eriştiğimizi görmek için kullanılır
# ROC altındaki alanı bulur alanın büyümesi bizim için olumludur










