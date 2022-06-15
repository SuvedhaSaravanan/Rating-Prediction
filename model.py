import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Zomato_df.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.head())

x = df.drop('rate', axis=1)
y = df['rate']
x_train, x_test, y_train, y_test  = train_test_split(x,y,test_size=0.3, random_state=10)

print(x_train.columns)
from sklearn.ensemble import ExtraTreesRegressor
ETM = ExtraTreesRegressor(n_estimators= 120)
ETM.fit(x_train, y_train)

y_predict = ETM.predict(x_test)
print(r2_score(y_test, y_predict))

import pickle 
pickle.dump(ETM, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict(x_test))
