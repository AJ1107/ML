#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[3]:


df=pd.read_csv('final_data_Hotel.csv',parse_dates=True, squeeze=True)


# In[4]:


df['Dates'] = pd.to_datetime(df['Dates'], format='%d-%m-%Y')
df.set_index('Dates', inplace=True)


data_to_decompose = df['Occperc']


result = seasonal_decompose(data_to_decompose, model='multiplicative', period=365)


# In[5]:


result


# In[6]:


plt.rcParams.update({'figure.figsize':(10,10)})
result.plot()
plt.show()


# In[7]:


result1 = seasonal_decompose(data_to_decompose, model='additive', period=365)


# In[8]:


plt.rcParams.update({'figure.figsize':(10,10)})
result1.plot()
plt.show()


# In[ ]:




