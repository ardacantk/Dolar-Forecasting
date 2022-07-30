# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:14:02 2022

@author:    Arda Can Tekküpeli
number:     201723024
Lesson:     EEE442_Artificial_Intelligence
Instructor: Assoc. Prof. Erdem Bilgili

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

"""
USD-TRY Data Frame with date index 
"""
df = pd.read_csv("USDTRY=X.csv")
del df["Open"]
del df["High"]
del df["Low"] 
del df["Adj Close"] 
del df["Volume"]


# Updating the header, change index with date.
df.columns=["Date","Sales"]
df.head()
df.describe()
df["Date"] = pd.to_datetime(df["Date"])
df.set_index('Date',inplace=True)



from pylab import rcParams
rcParams['figure.figsize'] = 15, 7
df.plot()
plt.title("USD-TRY Per Day Close Prices of Graph")
plt.show()


# P Value Control
from statsmodels.tsa.stattools import adfuller
test_result=adfuller(df['Sales'])

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
   
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary\n")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary\n")

adfuller_test(df['Sales'])

#  First difference and seasonal difference
df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(15)
df.head()

# P Value test with Seasonal First Difference
adfuller_test(df['Seasonal First Difference'].dropna())

# Seasonal First Difference plot
df['Seasonal First Difference'].plot()
plt.title("Seasonal First Difference Graph")
plt.show()

# Auto-correlation plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.title("Auto-Correlation")
plt.show()

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df['Seasonal First Difference'].iloc[16:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df['Seasonal First Difference'].iloc[16:],lags=40,ax=ax2)

# For non-seasonal data
#p=1, d=1, q=0 or 1
import statsmodels.api as sm
model = sm.tsa.arima.ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()

df['forecast']=model_fit.predict(start="2022-01-11", end="2022-05-12",dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,15))
results=model.fit()
df['forecast']=results.predict(start="2022-01-11" ,end="2022-05-12" ,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))

from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1] + DateOffset(weekday=x)for x in range(0,30)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

future_datest_df.tail()

future_df=pd.concat([df,future_datest_df])

future_df['forecast'] = results.predict(start ="2022-05-13" ,end = "2022-06-12", dynamic= True)
future_df[['Sales', 'forecast']].plot(figsize=(12, 8))

