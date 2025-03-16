#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('final_data_Hotel.csv',parse_dates=['Dates'])


# In[3]:


df


# In[5]:


df['Dates'] = pd.to_datetime(df['Dates'],format='%d-%m-%Y')


# In[6]:


df


# In[8]:


df['day_of_year'] = df['Dates'].dt.dayofyear


# In[12]:


df.tail(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import seaborn as sns


# In[6]:


df.sample(5)


# In[7]:


df[df['RoomSold']==90]['Dates']


# In[8]:


df=df.drop(1676)


# In[9]:


sns.countplot(df['WeekType'])


# In[10]:


sns.countplot(df['DayOfWeek'])


# In[11]:


plt.hist(df['RoomSold'])


# In[12]:


plt.hist(df['RoomSold'],bins=50)


# In[13]:


sns.distplot(df['RoomSold'])


# # here in this graph means what is the probability of 40 room was sold so its 0.025*100 = 25 %

# In[14]:


df['RoomSold'].skew()


# In[15]:


df['RoomSold'].max()


# In[16]:


df['RoomSold'].idxmin()
df.iloc[1495]


# In[17]:


sns.boxplot(df['RoomSold'])


# In[18]:


sns.scatterplot(df['RoomSold'],df['ADR'],hue=df['WeekType'])


# In[19]:


sns.scatterplot(df['ADR'],df['TotalRevenue'],hue=df['WeekType'])


# In[20]:


sns.scatterplot(df['RoomSold'],df['TotalRevenue'],hue=df['WeekType'])


# In[21]:


sns.scatterplot(df['RoomSold'],df['TotalRevenue'],hue=df['WeekType'],style=df['DayOfWeek'])


# # numerical numreical 

# In[22]:


plt.figure(figsize=(12,10))
sns.barplot(df['DayOfWeek'],df['RoomSold'])


# #  x in categorical  and numreical 

# In[23]:


sns.barplot(df['WeekType'],df['RoomSold'])


# # this showa avg room was sold

# In[24]:


plt.figure(figsize=(12,5))
sns.barplot(df['Dates'].dt.month,df['RoomSold'],hue=df['WeekType'])
plt.xlabel('month')


# In[25]:


sns.boxplot(df['DayOfWeek'],df['RoomSold'],hue=df['WeekType'])


# In[26]:


sns.distplot(df[df['WeekType']=='Weekday']['RoomSold'],hist=False,color='red')
sns.distplot(df[df['WeekType']=='Weekend']['RoomSold'],hist=False)


# In[27]:


sns.distplot(df[df['DayOfWeek']=='Fri']['RoomSold'],hist=False,color='red')
sns.distplot(df[df['DayOfWeek']=='Sat']['RoomSold'],hist=False)


# In[28]:


df[df['WeekType']=='weekday']['RoomSold']


# In[29]:


sns.pairplot(df)


# # lineplot numerical numrerical

# In[30]:


af=df.groupby((df['Dates'].dt.year)).sum()


# In[31]:


af


# In[32]:


af.describe()


# In[33]:


y=df.groupby(df['Dates'].dt.month).sum().reset_index()


# In[34]:


# df.groupby((df['Dates'].dt.year)& (df['Dates'].dt.year!=2024)).sum()
gf=df[df['Dates'].dt.year!=2024]
gf


# In[35]:


x=gf.groupby(df['Dates'].dt.year).sum().reset_index()


# In[36]:


x


# In[37]:


x.describe()


# In[38]:


sns.lineplot(y['Dates'],y['RoomSold'])


# In[39]:


df.groupby(df['DayOfWeek']).count()


# In[40]:


df.groupby(df['WeekType']).sum()


# In[41]:


df.groupby(df['WeekType']).count().reset_index()


# In[42]:


z=df.groupby(df['WeekType']).mean()


# In[43]:


z


# In[44]:


pd.crosstab(df['RoomSold'],df['WeekType'])


# In[45]:


sns.heatmap(pd.crosstab(df['RoomSold'],df['WeekType']))


# In[46]:


df.groupby(df['WeekType']).mean()['RoomSold']


# In[47]:


sns.clustermap(y)


# In[ ]:





# In[48]:


data=df


# In[49]:


data


# In[50]:


data['Month'] = data['Dates'].dt.month


# In[51]:


data.sample(5)


# In[52]:


data.pivot_table(values='RoomSold', index='Month', aggfunc='median')


# In[53]:


data.groupby(data['Month']).reset_index()


# # overall data

# In[54]:


sns.clustermap(data.pivot_table(values='RoomSold', index='Month', aggfunc='median'),cmap="coolwarm", standard_scale=1, col_cluster=False)


# In[55]:


data['year']=df['Dates'].dt.year


# In[56]:


only_2022=data[data['year']==2022]


# In[57]:


only_2022


# In[58]:


only_2022.pivot_table(index='Month',values='RoomSold', aggfunc='median')


# # 2022 clustermap

# In[59]:


sns.clustermap(only_2022.pivot_table(index='Month',values='RoomSold', aggfunc='median'),cmap="coolwarm", standard_scale=1, col_cluster=False)


# In[60]:


plt.figure(figsize=(12,5))
sns.lineplot(df['Dates'].dt.year,df['RoomSold'])
plt.xticks([2014,2015,2016,2017,2018,2019,2021,2022,2023])
plt.show()


# In[61]:



plt.figure(figsize=(12,5))
sns.lineplot(df['Dates'].dt.year,df['ADR'])
plt.xticks([2014,2015,2016,2017,2018,2019,2021,2022,2023])
plt.show()


# In[62]:


sns.barplot(df['Season'],df['RoomSold'])


# #Spring: March (3), April (4), May (5)
# #Summer: June (6), July (7), August (8)
# #Fall (Autumn): September (9), October (10), November (11)
# #Winter: December (12), January (1), February (2)

# In[63]:


sns.barplot(df['Season'],df['ADR'])


# In[64]:


sns.boxplot(df['Season'],df['RoomSold'])


# In[65]:


sns.barplot(df['Season'],df['Update Revenue'])


# In[66]:


sns.kdeplot(df['RoomSold'])


# In[67]:


sns.kdeplot(df['ADR'])


# In[68]:


df['ADR'].skew()


# # this show distrubation

# In[69]:


sns.kdeplot(df['Dates'].dt.year,df['RoomSold'])


# In[70]:


df['WeekType'].value_counts()


# In[71]:


df['DayOfWeek'].value_counts()


# In[72]:


plt.figure(figsize=(12,5))
sns.distplot(df[df['DayOfWeek']=='Mon']['RoomSold'],hist=False,color='red')
sns.distplot(df[df['DayOfWeek']=='Tue']['RoomSold'],hist=False,color='yellow')
sns.distplot(df[df['DayOfWeek']=='Wed']['RoomSold'],hist=False,color='blue')
sns.distplot(df[df['DayOfWeek']=='Thu']['RoomSold'],hist=False,color='black')
sns.distplot(df[df['DayOfWeek']=='Fri']['RoomSold'],hist=False,color='violet')
sns.distplot(df[df['DayOfWeek']=='Sat']['RoomSold'],hist=False,color='green')
sns.distplot(df[df['DayOfWeek']=='Sun']['RoomSold'],hist=False,color='orange')
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90])
plt.show()


# In[73]:


import numpy as np
grouped_data = df.groupby(['DayOfWeek', 'RoomSold']).size().reset_index(name='Count')


pivot_data = grouped_data.pivot(index='RoomSold', columns='DayOfWeek', values='Count').fillna(0)


pivot_data_prob = pivot_data.div(pivot_data.sum(axis=1), axis=0)


pivot_data_max_prob = pivot_data_prob.idxmax(axis=1).reset_index()
pivot_data_max_prob.columns = ['RoomSold', 'MostLikelyDay']
pivot_data_max_prob['MaxProbability'] = pivot_data_prob.max(axis=1).values


range_step = 10


pivot_data_max_prob['RoomSoldRange'] = pd.cut(pivot_data_max_prob['RoomSold'], 
                                              bins=np.arange(0, pivot_data_max_prob['RoomSold'].max() + range_step, range_step), 
                                              right=False)


range_summary = pivot_data_max_prob.groupby('RoomSoldRange').agg(
    MostLikelyDay=('MostLikelyDay', lambda x: x.mode()[0]),
    AvgProbability=('MaxProbability', 'mean') 
).reset_index()

# Display the range summary
print(range_summary)


# In[74]:


import numpy as np
grouped_data = df.groupby(['DayOfWeek', 'RoomSold']).size().reset_index(name='Count')


pivot_data = grouped_data.pivot(index='RoomSold', columns='DayOfWeek', values='Count').fillna(0)


pivot_data_prob = pivot_data.div(pivot_data.sum(axis=1), axis=0)


pivot_data_max_prob = pivot_data_prob.idxmax(axis=1).reset_index()
pivot_data_max_prob.columns = ['RoomSold', 'MostLikelyDay']
pivot_data_max_prob['MaxProbability'] = pivot_data_prob.max(axis=1).values


range_step = 20
pivot_data_max_prob['RoomSoldRange'] = pd.cut(pivot_data_max_prob['RoomSold'], 
                                              bins=np.arange(0, pivot_data_max_prob['RoomSold'].max() + range_step, range_step), 
                                              right=False)


range_summary = pivot_data_max_prob.groupby('RoomSoldRange').agg(
    MostLikelyDay=('MostLikelyDay', lambda x: x.mode()[0]),
    AvgProbability=('MaxProbability', 'mean') 
).reset_index()
print(range_summary)


# In[75]:


bins = np.arange(0, df['RoomSold'].max() + 10, 10)
df['RoomSoldRange'] = pd.cut(df['RoomSold'], bins=bins, right=False)
range_day_distribution = df.groupby(['RoomSoldRange', 'DayOfWeek']).size().unstack(fill_value=0)
range_day_probabilities = range_day_distribution.div(range_day_distribution.sum(axis=1), axis=0)
range_day_probabilities = range_day_probabilities.reset_index()
print(range_day_probabilities)


# # My analysis 

# # 1. 0-10 sun

# # 2.10-20 sun
# # 3.20-30 sun:->0.26
# # 4.30-40 sun:-> 0.18 wed:-> 0.17  thu and mon:->0.15
# # 5.40-50 Mon:->0.19 fri and sat:-> 0.11 sun and thu:->0.12
# # 6.50-60 thu:->0.18 fri and wed :-> 0.16 Tue:->0.15
# # 7.60-70 fri:->0.20 and  thu:->0.1966  sat:->0.15 other same 
# # 8.70-80 sat:->0.24 fri:->0.20 thu tue wed same 
# # 8.80-90 sat:->0.35 fri:->0.27

# In[76]:


bins = np.arange(0, 100, 10) 
df['RoomSoldRange'] = pd.cut(df['RoomSold'], bins=bins, right=False)
season_range_day_distribution = df.groupby(['Season', 'RoomSoldRange', 'DayOfWeek']).size().unstack(fill_value=0)
season_range_day_probabilities = season_range_day_distribution.div(season_range_day_distribution.sum(axis=1), axis=0)
season_range_day_probabilities.reset_index()


# In[77]:


winter_data= season_range_day_probabilities.query("Season == 'Winter'").droplevel(0).reset_index()
winter_data


# In[78]:


summer_data = season_range_day_probabilities.query("Season == 'Summer'").droplevel(0).reset_index()
summer_data


# In[79]:


spring_data = season_range_day_probabilities.query("Season == 'Spring'").droplevel(0).reset_index()
spring_data


# In[80]:


fall_data = season_range_day_probabilities.query("Season == 'Fall'").droplevel(0).reset_index()
fall_data


# In[81]:


fig, ax = plt.subplots(figsize=(14, 8))
width = 0.1  
days = summer_data.columns[1:]  
ind = np.arange(len(summer_data['RoomSoldRange']))

for i, day in enumerate(days):
    ax.bar(ind + i * width, summer_data[day], width, label=day)
ax.set_xlabel('RoomSold Range')
ax.set_ylabel('Probability')
ax.set_title('Probability of Each Day of the Week in Summer ')
ax.set_xticks(ind + width * len(days) / 2)
ax.set_xticklabels([str(x) for x in summer_data['RoomSoldRange']])
ax.legend(title='Day of the Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[82]:


spring_data = season_range_day_probabilities.query("Season == 'Spring'").droplevel(0).reset_index()


fig, ax = plt.subplots(figsize=(14, 8))
width = 0.1  
days = spring_data.columns[1:]  
ind = np.arange(len(spring_data['RoomSoldRange']))
for i, day in enumerate(days):
    ax.bar(ind + i * width, spring_data[day], width, label=day)

ax.set_xlabel('RoomSold Range')
ax.set_ylabel('Probability')
ax.set_title('Probability of Each Day of the Week in  Spring')
ax.set_xticks(ind + width * len(days) / 2)
ax.set_xticklabels([str(x) for x in spring_data['RoomSoldRange']])
ax.legend(title='Day of the Week')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[83]:


spring_data = season_range_day_probabilities.query("Season == 'Winter'").droplevel(0).reset_index()


fig, ax = plt.subplots(figsize=(14, 8))
width = 0.1  
days = spring_data.columns[1:]  
ind = np.arange(len(spring_data['RoomSoldRange']))
for i, day in enumerate(days):
    ax.bar(ind + i * width, spring_data[day], width, label=day)

ax.set_xlabel('RoomSold Range')
ax.set_ylabel('Probability')
ax.set_title('Probability of Each Day of the Week in  Winter')
ax.set_xticks(ind + width * len(days) / 2)
ax.set_xticklabels([str(x) for x in spring_data['RoomSoldRange']])
ax.legend(title='Day of the Week')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[84]:


spring_data = season_range_day_probabilities.query("Season == 'Fall'").droplevel(0).reset_index()


fig, ax = plt.subplots(figsize=(14, 8))
width = 0.1  
days = spring_data.columns[1:]  
ind = np.arange(len(spring_data['RoomSoldRange']))
for i, day in enumerate(days):
    ax.bar(ind + i * width, spring_data[day], width, label=day)

ax.set_xlabel('RoomSold Range')
ax.set_ylabel('Probability')
ax.set_title('Probability of Each Day of the Week in Fall')
ax.set_xticks(ind + width * len(days) / 2)
ax.set_xticklabels([str(x) for x in spring_data['RoomSoldRange']])
ax.legend(title='Day of the Week')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[85]:


bins = np.arange(0, df['RoomSold'].max() + 10, 10)
df['RoomSoldRange'] = pd.cut(df['RoomSold'], bins=bins, right=False)
range_day_distribution = df.groupby(['RoomSoldRange', 'DayOfWeek']).size().unstack(fill_value=0)
range_day_probabilities = range_day_distribution.div(range_day_distribution.sum(axis=1), axis=0)
range_day_probabilities = range_day_probabilities.reset_index()


fig, ax = plt.subplots(figsize=(14, 8))
width = 0.1  
days = range_day_probabilities.columns[1:]  
indices = np.arange(len(range_day_probabilities['RoomSoldRange']))  

for i, day in enumerate(days):
    ax.bar(indices + i * width, range_day_probabilities[day], width, label=day)

ax.set_xlabel('RoomSold Range')
ax.set_ylabel('Probability')
ax.set_title('Day of the Week Probabilities Across RoomSold Ranges')
ax.set_xticks(indices + width * len(days) / 2)
ax.set_xticklabels(range_day_probabilities['RoomSoldRange'].astype(str), rotation=45)
ax.legend(title='Day of the Week', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# In[86]:


pip install prophet


# In[87]:


df_prophet['ds'] = pd.to_datetime(df['Dates'])
df_prophet['y'] = df['RoomSold']
# Add additional regressors
df_prophet['month'] = df['Month']
df_prophet['year'] = df['Year']
df_prophet['season'] = df['Season'].astype('category').cat.codes  # Convert season to numerical codes


# In[ ]:


from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[ ]:


df_prophet = pd.DataFrame()


df_prophet['ds'] = pd.to_datetime(df['Dates'])
df_prophet['y'] = df['RoomSold']


df['month'] = df['Month']
df['year'] = df['Year']
df['season'] = df['Season'].astype('category').cat.codes  


# In[ ]:


df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


# In[88]:


model = Prophet()
model.add_regressor('month')
model.add_regressor('year')
model.add_regressor('season')
model.fit(df_train[['ds', 'y', 'month', 'year', 'season']])


# In[89]:


season_day_probabilities = {
    ('Winter', 'Mon'): 0.1,  
   


# In[90]:


season_data_combined = pd.concat([summer_data, winter_data, spring_data, fall_data], ignore_index=True)


# In[91]:


season_data_combined 


# In[92]:


season_data_combined = season_data_combined.drop('DaysOfWeek', axis=1)


# In[93]:


print(season_data_combined.columns)
season_data_combined.rename(name={'DayOfWeek':'index'})


# In[94]:


season_data_combined.columns.name = 'Index'


# In[95]:


season_data_combined


# In[96]:


def apply_probabilities(row):
    return season_range_day_probabilities.get((row['Season'], row['DayOfWeek']), 0)  

# Add a new column for probabilities
df['sale_probability'] = df.apply(apply_probabilities, axis=1)


# In[97]:


df.drop('sale_probability',axis=1)


# In[98]:


df


# In[99]:


df


# In[100]:


plt.plot(df['Season'],df['RoomSold'])


# In[101]:


season_roomsold = df.groupby('Season')['RoomSold'].mean().reset_index()

# Plotting
plt.plot(season_roomsold['Season'], season_roomsold['RoomSold'])
plt.xlabel('Season')
plt.ylabel('RoomSold')
plt.title('Room Sold by Season')
plt.xticks(rotation=45)  # Rotate labels if necessary
plt.show()


# In[102]:


plt.plot(df['Dates'],df['RoomSold'])


# In[103]:


df.describe()


# In[104]:



plt.figure(figsize=(20,7))
df.RoomSold.plot()
plt.show()


# In[105]:


plt.figure(figsize=(20,7))
df.ADR.plot()
plt.show()


# In[106]:


plt.plot(df['Season'],df['RoomSold'])


# In[107]:


sns.histplot(x=df['Season'],y=df['RoomSold'])


# In[108]:


df


# In[109]:


plt.scatter(df['RoomSold'],df['ADR'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




