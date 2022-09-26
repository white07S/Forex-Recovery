import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json 
import os
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler


df = yf.download('EURUSD=X',period='2y',interval='1d').reset_index()


plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
# plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('avg Price',fontsize=18)
plt.show()

# high_prices = df.loc[:,'High'].as_matrix()
high_prices = df.as_matrix(df.loc[:,"High"])
# low_prices = df.loc[:,'Low'].as_matrix()
low_prices = df.as_matrix(df.loc[:,"Low"])
avg_prices = (high_prices+low_prices)/2.0

train = avg_prices[:11000]
test = avg_prices[11000:]
print(len(avg_prices))
