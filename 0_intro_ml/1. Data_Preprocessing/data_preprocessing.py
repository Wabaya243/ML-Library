import numpy as np	
import matplotlib.pyplot as plt
import pandas as pd


#importer le dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#gerer les valeurs manquantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#gerer les variables catégoriques
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#diviser le dataset en training set et test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train) #on fit le training set car on veut que le training set soit influencé par le training set
x_test = sc.transform(x_test) #on ne fit pas le test set car on ne veut pas que le test set soit influencé par le training set

