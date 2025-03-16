#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import numpy as np


# In[3]:


from statsmodels.tsa.arima.model import ARIMA


# In[4]:


df=pd.read_csv('final_data_with_levels.csv',parse_dates=True)
df['Dates'] = pd.to_datetime(df['Dates'], format='%d-%m-%Y')


# In[5]:


df


# In[6]:


df['Level'].value_counts()


# In[ ]:


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import itertools
import numpy as np

# Load your data


# Define your parameter ranges
p_values = [1, 2, 3, 4, 5, 6]
d_values = [0, 1, 2]
q_values = [1, 2, 3, 4, 5, 6]
pdq_combinations = list(itertools.product(p_values, d_values, q_values))

best_aic = np.inf
best_pdq = None

for combination in pdq_combinations:
    try:
        model = ARIMA(df['Occperc'], order=combination)
        model_fit = model.fit()
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_pdq = combination
    except Exception as e:
        continue

print(f"Best AIC: {best_aic}, Best PDQ: {best_pdq}")# 6 ,0 ,6 


# In[7]:


df.info()


# In[8]:


df['Level'] = pd.to_numeric(df['Level'], errors='coerce') 


# In[9]:


df.set_index('Dates', inplace=True)


# In[10]:


df.sort_index(inplace=True)


# In[11]:


df


# In[12]:


df['Level'].fillna(1, inplace=True)


# In[13]:


train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]


# In[14]:


train_occperc = train['Occperc']
test_occperc = test['Occperc']
train_level = train[['Level']]
test_level = test[['Level']]


# In[15]:


model = ARIMA(train_occperc, exog=train_level, order=(6,0,6))
model_fit = model.fit()


# In[30]:


predictions = model_fit.forecast(steps=len(test), exog=test_level)


# In[31]:


predictions_series = pd.Series(predictions.values, index=test.index)


# In[32]:


comparison = pd.DataFrame({'actual': test_occperc, 'predicted': predictions_series})


# In[33]:


print(comparison)


# In[34]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[35]:


mse = mean_squared_error(comparison['actual'], comparison['predicted'])


# In[36]:


mse


# In[37]:


mae = mean_absolute_error(comparison['actual'], comparison['predicted'])


# In[38]:


mae


# In[39]:


mape = np.mean(np.abs((comparison['actual'] - comparison['predicted']) / comparison['actual'])) * 100

# Define accuracy as 100% - MAPE
accuracy_percentage = 100 - mape


# In[40]:


accuracy_percentage


# In[41]:


comparison.head(20)


# In[ ]:





# In[ ]:





# In[ ]:




