import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Charger les données (remplacez 'data.csv' par le vrai nom de votre fichier)
fuelConsumption_df = pd.read_csv('FuelConsumption.csv')
print(fuelConsumption_df.head())
fuelConsumption_df.info()

# Sélectionner les attributs et la variable cible
features = ['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']
X = fuelConsumption_df[features]
y = fuelConsumption_df['CO2EMISSIONS'] 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15 , shuffle=False,random_state = 0)

model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False),StandardScaler(),  LinearRegression())
model.fit(x_train, y_train)

import pickle

filename = 'model.pickle'
pickle.dump(model, open(filename, 'wb'))

print("Modèle entraîné et sauvegardé dans model.pickle")
