#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np


# In[ ]:


from statsmodels.tsa.arima.model import ARIMA


# In[ ]:



df=pd.read_csv('final_data_with_levels.csv',parse_dates=True)
df['Dates'] = pd.to_datetime(df['Dates'], format='%d-%m-%Y')


# In[ ]:


df


# In[ ]:


ts=df[['Dates','Occperc']]
plt.plot(ts)
ts.set_index('Dates', inplace=True)
ts


# In[ ]:


from statsmodels.tsa.stattools import acf, pacf  

lag_acf = acf(ts, nlags=20)
lag_pacf = pacf(ts, nlags=20, method='ols')


# In[ ]:


#Plot ACF:    
plt.figure(figsize=(12,5))
plt.subplot(121)    
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


# In[ ]:


X=ts.values
train, test = X[0:2300], X[2300:len(X)]
X


# In[ ]:


train


# In[ ]:


import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


data = pd.read_csv('final_data_Hotel.csv')
data['Dates'] = pd.to_datetime(data['Dates'], format='%d-%m-%Y')
data.set_index('Dates', inplace=True)

# Define ranges for p, q, and d
p_values = [1,2,3,4,5,6]
d_values = [0,1,2]
q_values = [1,2,3,4,5,6]



best_rmse, best_p, best_d, best_q = np.inf, None, None, None
history = [x for x in train]
# make predictions
predictions = list()
# Perform grid search
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                for t in range(len(test)):
                    # Fit the model
                    model = ARIMA(history, order=order)
                    model_fit = model.fit()
                    yhat = model_fit.forecast()[0]
                    predictions.append(yhat)
                    history.append(test[t])

              # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(test, predictions))

              # Update best RMSE and parameter values
                if rmse < best_rmse:
                     best_rmse, best_p, best_d, best_q = rmse, p, d, q

            except:
                continue

print(f"Best RMSE: {best_rmse}")
print(f"Best p: {best_p}")
print(f"Best d: {best_d}")
print(f"Best q: {best_q}")


# In[ ]:


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import itertools
import numpy as np

# Load your data
data = pd.read_csv('final_data_Hotel.csv')
data['Dates'] = pd.to_datetime(data['Dates'], format='%d-%m-%Y')
data.set_index('Dates', inplace=True)

# Define your parameter ranges
p_values = [1, 2, 3, 4, 5, 6]
d_values = [0, 1, 2]
q_values = [1, 2, 3, 4, 5, 6]
pdq_combinations = list(itertools.product(p_values, d_values, q_values))

best_aic = np.inf
best_pdq = None

for combination in pdq_combinations:
    try:
        model = ARIMA(data['Occperc'], order=combination)
        model_fit = model.fit()
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_pdq = combination
    except Exception as e:
        continue

print(f"Best AIC: {best_aic}, Best PDQ: {best_pdq}")


# In[ ]:


data


# In[ ]:


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# In[ ]:


history = [x for x in train]
predictions = list()
#test.reset_index()
for t in range(len(test)):
    try:
        model = ARIMA(history, order=(6,0,6))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    except (ValueError, LinAlgError):
        pass
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
rmse = mean_squared_error(test, predictions)**0.5
print('Test MSE: %.3f' % rmse)


from math import sqrt
rms = sqrt(mean_squared_error(test, predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




