import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error , r2_score

from sklearn.model_selection import train_test_split

#导入数据
data= pd.read_csv("AirQualityUCI.csv",sep=';', decimal=",")

data.dropna(how='all',inplace=True)#去掉空行
data.dropna(thresh=10,axis=0,inplace=True)#rh为空
data.dropna(axis=1, how= 'all', inplace=True)
data = data.replace(-200, np.nan)
print(data.isnull().sum())
data.drop('NMHC(GT)',axis=1,inplace=True)
data['month']=data['Date'].apply(lambda x: int(x.split('/')[1]))
data['Time']=data['Time'].apply(lambda x: int(x.split('.')[0]))
data['Date']=pd.to_datetime(data.Date, format='%d/%m/%Y')

print(data.describe())
for str in ['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','CO(GT)','C6H6(GT)','NOx(GT)','NO2(GT)','T','AH',"RH"]:
    data[str] = data[str].fillna(data.groupby(['Time','month'])[str].transform('mean'))
print(data.isnull().sum())
print(data.Date.isnull().values.any())
data['CO(GT)']=data['CO(GT)'].fillna(data.groupby(['Time'])['CO(GT)'].transform('mean'))
data['NOx(GT)']=data['NOx(GT)'].fillna(data.groupby(['Time'])['NOx(GT)'].transform('mean'))
data['NO2(GT)']=data['NO2(GT)'].fillna(data.groupby(['Time'])['NO2(GT)'].transform('mean'))

X=data[['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','CO(GT)','C6H6(GT)','NOx(GT)','NO2(GT)','T','month']]
#X=data[['PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','NOx(GT)','NO2(GT)','T','month']]

X_train , X_test , y_train ,y_test = train_test_split(X,data['RH'],test_size=0.2,random_state=42)
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
print(regr.score(X_train,y_train))#r2 score
print(regr.score(X_test,y_test))#r2 score
print('Coefficients: \n', regr.coef_)
preditct_test=regr.predict(X_test)
print('均方误差: %.3f' % np.sqrt(mean_squared_error(y_test,preditct_test)))
plt.scatter(preditct_test  ,y_test,color ='green')
plt.axis([0, 90, 0, 90])
plt.xlabel('predict')
plt.ylabel('true value')
plt.show()