#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from prophet import Prophet


# In[2]:


data = pd.read_csv('final_data_Hotel.csv')  # Update the path to your file
data['Dates'] = pd.to_datetime(data['Dates'])  # Convert your dates to datetime objects
data.rename(columns={'Dates': 'ds', 'Occperc': 'y'}, inplace=True)  # Prophet requires column names to be 'ds' and 'y'

# Trim data to relevant columns
data = data[['ds', 'y']]


# In[3]:


data


# In[4]:


data['y_orig'] = data['y']


# In[6]:


import numpy as np


# In[7]:


data['y'] = np.log(data['y'])


# In[10]:


model=Prophet()
model.add_country_holidays(country_name='US')


# In[11]:


model.fit(data)


# In[12]:


future_data = model.make_future_dataframe(periods=10, freq = 'D')


# In[14]:


forecast_data = model.predict(future_data)


# In[15]:


forecast_data.tail(10)


# In[16]:


forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)


# In[17]:


model.plot(forecast_data)


# In[18]:


model.plot_components(forecast_data)


# In[19]:


forecast_data_orig = forecast_data # make sure we save the original forecast data
forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])


# In[20]:


model.plot(forecast_data_orig)


# In[21]:


data['y_log']=data['y'] #copy the log-transformed data to another column
data['y']=data['y_orig']


# In[22]:


final_df = pd.DataFrame(forecast_data_orig)


# In[23]:


import plotly.graph_objs as go
import plotly.offline as py


# In[25]:


#Plot predicted and actual line graph with X=dates, Y=Outbound
actual_chart = go.Scatter(y=data["y_orig"], name= 'Actual')
predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower], image_width=400, image_height=400)
#py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower], filename = 'templates/' +'filename.html', auto_open=False)


# In[26]:


forecast_data_orig


# In[ ]:




