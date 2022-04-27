import numpy as np 
import pandas as pd 
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("Ann.csv") 

X= df.loc[:, df.columns != "FOS"]
y = df["FOS"]

GBR = GradientBoostingRegressor().fit(X,y)
GBR.score(X,y)


import joblib 

joblib.dump(GBR, "GBR.pkl") 
