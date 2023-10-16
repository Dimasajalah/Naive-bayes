# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 17:43:00 2022

@author: DIMAS ANGGORO SAKTI
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer

direktori = "C:/Users/DIMAS ANGGORO SAKTI/Dropbox/My PC (LAPTOP-FMDVD92V)/Downloads/C_2110511115_Dimas Anggoro sakti naive bayes/folder tugas/hepatitis.data"

header = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA',
         'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPLABLE', 'SPIDERS', 'ASCITES', 
         'VARICES', 'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']

# membaca data dengan library panda 
data = pd.read_csv(direktori, names=header)

# mengganti '?' dengan nan
data.replace('?', np.nan, inplace = True)

# panggil nilai dataset
arrayData = data.values

# pisah input dan output
x_data = arrayData[:,1:] # inputnya adalah kolom ke-1, 2, 3, dst...
y_data = arrayData[:,0] # outputnya adalah kolom ke 0

# imputasi mean
imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

# simpan hasil imputasi ke dalam variable X_data
X_data = imp.fit_transform(x_data)

# mengubah kategori dari String ke numerik
le = preprocessing.LabelEncoder()
y_data = le.fit_transform(y_data)

# membagi data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)

# gaussian naive bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# mendapatkan hasil prediksi dari data testing
y_pred = gnb.predict(X_test)

# Menampilkan presentasi error prediksi
error = ((y_test!=y_pred).sum()/len(y_pred))*100
print("Error prediksi = %.2f" %error, "%")

# menampilkan presentasi akurasi
akurasi = 100 - error
print("Akurasi = %.2f" %akurasi, "%")