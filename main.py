import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
'''dataset=pd.read_csv('50_Startup.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values
from sklearn.linear_model import LinearRegression
regg=LinearRegression()
regg.fit(X,y)
pickle.dump(regg,open('model.pkl','wb'))
'''

model=pickle.load(open('logistic.pkl','rb'))
print(model.predict([[0,0,1,14,11,11,]]))