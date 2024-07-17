import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\decision trees\\aylara_gore_satis.csv")
months=datas.iloc[:,0].values.reshape(-1,1)
sales=datas.iloc[:,1].values.reshape(-1,1)
dtree_reg=DecisionTreeRegressor()
dtree_reg.fit(months,sales)
predict=dtree_reg.predict(months)
plt.scatter(months,sales)
plt.plot(months,predict)
plt.show()








