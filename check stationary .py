#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('final_data_Hotel.csv',parse_dates=True, squeeze=True)


# In[3]:


df['Dates'] = pd.to_datetime(df['Dates'], format='%d-%m-%Y')
df.set_index('Dates', inplace=True)


# In[4]:


from statsmodels.tsa.stattools import adfuller


# In[5]:


def adfuller_test(ADR):
    result=adfuller(ADR)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    


# In[6]:


adfuller_test(df['Occperc'])


# In[7]:


adfuller_test(df['RoomSold'])


# In[8]:


adfuller_test(df['ADR'])


# In[ ]:




