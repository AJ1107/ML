#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 69 Use load_iris from sklearn.datasets to load the Iris dataset. initialize the LDA object with
# n_components=2 to reduce the dataset to 2. use matplotlib to plot the LDA-reduced dataset in 2D

# lda is a classification graph its help to things into catorize 
# c=y assigns a color to each point based on its class (0, 1, 2).

import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
import  matplotlib.pyplot as  plt 
from sklearn.preprocessing import StandardScaler

df= load_iris()
x= df.data
y= df.target

lda= LDA(n_components=2)
x_lda=lda.fit_transform(x,y)

plt.scatter(x_lda[:,0],x_lda[:,1],c=y,cmap="rainbow",edgecolor='k')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.title('LDA on Iris Dataset')
plt.colorbar(label="Flower Type")
plt.show()


# In[2]:


# 68. Use load_digits() from sklearn.datasets to load the hand written numbers dataset. initialize the
# LDA object with two component to reduce the dataset. Use matplotlib to plot the LDA-reduced
# dataset

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_digits
import  matplotlib.pyplot as  plt 

df= load_digits()
x= df.data
y=df.target



lda= LDA(n_components=2)
x_lda=lda.fit_transform(x,y)

plt.scatter(x_lda[:,0],x_lda[:,1],c=y)
plt.show()


# In[3]:


# 67. Use breast cancer from sklearn.datasets to load the cancer dataset. initialize the LDA object with
# one component to reduce the dataset to 1D. use matplotlib to plot the LDA-reduced dataset.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_breast_cancer
import  matplotlib.pyplot as  plt 
import numpy as np

df=load_breast_cancer()
x= df.data
y=df.target

lda= LDA(n_components=1)
x_lda=lda.fit_transform(x,y)

plt.figure(figsize=(12,5))
plt.scatter(x_lda,np.zeros_like(x_lda),c=y)
plt.yticks([]) # for removing y axis 
plt.show()


# In[4]:


# 66. Use load_wine from sklearn.datasets to load the wine dataset. initialize the LDA object with
# n_components=2 to reduce the dataset to 2. use matplotlib to plot the LDA-reduced dataset in 2D
# represented by a different color

from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import  matplotlib.pyplot as  plt 

df=load_wine()
x= df.data
y=df.target

lda=LDA(n_components=2)
x_lda=lda.fit_transform(x,y)

plt.scatter(x_lda[:,0],x_lda[:,1],c=y)


# In[5]:


# 65. Use load_iris from sklearn.datasets to load the Iris dataset. initialize the PCA object with
# n_components=2 to reduce the dataset to 2. use matplotlib to plot the PCA-reduced dataset in 2D.

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import  matplotlib.pyplot as  plt 
from sklearn.preprocessing import StandardScaler

df= load_iris()
x= df.data
y= df.target

s=StandardScaler()
x=s.fit_transform(x)

pca=PCA(n_components=2)
x_pca=pca.fit_transform(x)

plt.scatter(x_pca[:,0],x_pca[:,1],c=y)


# In[6]:


# 64. Create synthetic Dataset using make_blobs with 3 features and 3 centers initialize the PCA object
# with n_components=2 to reduce the dataset to 2 dimensions. use matplotlib to plot the original 3D
# dataset and the PCA-reduced 2D dataset side by side

from sklearn.decomposition import PCA
import  matplotlib.pyplot as  plt 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

x,y=make_blobs(n_samples=100,n_features=3,centers=3)

s=StandardScaler()
x_scaled=s.fit_transform(x)

pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled)


fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(1, 2, 1, projection='3d') 
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap='viridis', marker='o')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Original 3D Data')

# Plot PCA Reduced 2D Data
plt.subplot(1, 2, 2)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Reduced 2D Data')

plt.show()


# In[7]:


# 63. Write a python snippet to demonstrate use of PCA on dataset of your choice. Also print explained
# variance of all principle components.
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import  matplotlib.pyplot as  plt 
from sklearn.preprocessing import StandardScaler

df= load_iris()
x= df.data
y= df.target

s=StandardScaler()
x=s.fit_transform(x)

pca=PCA(n_components=2)
x_pca=pca.fit_transform(x)
print(pca.explained_variance_ratio_)
plt.scatter(x_pca[:,0],x_pca[:,1],c=y)


# varince means
# 75% of the information is captured by PC1
# 20% of the information is captured by PC2
# 95% of the original data’s information is retained!


# In[8]:


# 62. Write a python snippet to demonstrate use of PCA on 100 samples with 3 features generated
# randomly.
from sklearn.decomposition import PCA
import numpy as np
import  matplotlib.pyplot as  plt 

# np.random.seed(42) ensures that when you generate random numbers, 
# you get the same numbers every time you run the code.
np.random.seed(32)
x= np.random.rand(100,3)

s=StandardScaler()
x=s.fit_transform(x)

pca=PCA(n_components=2)
x_pca=pca.fit_transform(x)

plt.scatter(x_pca[:,0],x_pca[:,1])
plt.show()


# In[42]:


# 61.Consider a dataset HealthData.csv with features such as age, bmi, blood_pressure, and a target
# column health_score. Write a Python code to fit a Lasso regression model using all available
# features to predict health_score. Use the complete dataset to train the model. Plot the coefficients
# of the features and create a residual plot.

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import  matplotlib.pyplot as  plt 
import seaborn as sns 

df= pd.read_csv("HealthData.csv")

x= df.drop(columns=['health_score'])
y= df['health_score']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

ls= Lasso(alpha=0.1)
ls.fit(x_train,y_train)

y_pred= ls.predict(x_test)

print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(r2_score(y_test,y_pred)) # for r2 score first test then predict

plt.figure(figsize=(10,5))
plt.bar(x.columns,ls.coef_)
plt.show()

# residual plot 
res=y_test-y_pred
sns.residplot(x=y_pred,y=res,lowess=True,line_kws={'color':'red'})

plt.show()


# In[41]:


# 60. Consider a dataset WineQuality.csv with various chemical properties of wine and a target column
# quality. Write a Python code to fit a Lasso regression model using all available features to predict
# quality. Use the complete dataset to train the model. Plot the coefficients of the features and create
# a residual plot.

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import  matplotlib.pyplot as  plt 
import seaborn as sns 

df=pd.read_csv('WineQuality.csv')
x= df.drop(columns=['quality'])
y= df['quality']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

ls=Lasso(alpha=0.1)
ls.fit(x_train,y_train)

y_pred= ls.predict(x_test)

r2=r2_score(y_test,y_pred) # for r2 score first test then predict
print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(20,10))
plt.bar(x.columns,ls.coef_)
plt.show()

res=y_test-y_pred
sns.residplot(x=y_pred,y=res,lowess=True)
plt.show()

df.info()


# In[45]:


# 59. Consider a dataset Diabetes.csv with various medical attributes and a target column
# diabetes_progression. Write a Python code to fit a Ridge regression model using all available
# features to predict diabetes_progression. Use the complete dataset to train the model. Plot the
# coefficients of the features and create a residual plot.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import  matplotlib.pyplot as  plt 
import seaborn as sns 



df=pd.read_csv('Diabetes (1).csv')
x=df.drop(columns=['DiabetesPedigreeFunction'])
y=df['DiabetesPedigreeFunction']


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

ls=Lasso(alpha=0.1)
ls.fit(x_train,y_train)

y_pred= ls.predict(x_test)

r2=r2_score(y_test,y_pred) # for r2 score first test then predict
print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(20,10))
plt.bar(x.columns,ls.coef_)
plt.show()

res=y_test-y_pred
sns.residplot(x=y_pred,y=res,lowess=True)
plt.show()


# In[48]:


# 58. Consider a dataset HousingData.csv with columns num_rooms, area, age, and a target column price.
# Write a Python code to fit a Lasso regression model using num_rooms, area, and age as independent
# variables to predict price. Use the complete dataset to train the model. Plot the coefficients of the
# features and create a residual plot.
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import  matplotlib.pyplot as  plt 
import seaborn as sns 


df=pd.read_csv('HousingData.csv')
x=df[['Bedrooms','SqFt','Offers']]
y=df['Price']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

ls=Lasso(alpha=0.1)
ls.fit(x_train,y_train)

y_pred= ls.predict(x_test)

r2=r2_score(y_test,y_pred) # for r2 score first test then predict
print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(20,10))
plt.bar(x.columns,ls.coef_)
plt.show()

res=y_test-y_pred
sns.residplot(x=y_pred,y=res,lowess=True)
plt.show()


# In[62]:


# 57. Consider a dataset EmployeeData.csv with features such as years_of_experience, education_level,
# and a target column salary. Write a Python code to fit a linear regression model to predict salary
# based on the features and evaluate the model using Mean Squared Error (MSE), R-squared (R²), and
# Mean Absolute Error (MAE)
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv('Salary.csv')
df=df.dropna()
df= pd.get_dummies(df,drop_first=True)

x=df.drop(columns=['Salary'])
y=df['Salary']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)


m=LinearRegression()
m.fit(x_train,y_train)

y_pred= m.predict(x_test)
r2=r2_score(y_test,y_pred) # for r2 score first test then predict
print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(f"R² Score: {r2:.2f}")


# In[67]:


# 56. Consider a dataset HouseData.csv with features such as num_rooms, square_feet, and location, and
# a target column house_price. Write a Python code to fit a linear regression model to predict
# house_price based on the features and evaluate the model using Mean Squared Error (MSE), R², and
# Mean Absolute Error (MAE).
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split



df=pd.read_csv('HouseData.csv')

x=df[['Bedrooms','SqFt','Offers']]
y=df['Price']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

ls=LinearRegression()
ls.fit(x_train,y_train)

y_pred= ls.predict(x_test)

r2=r2_score(y_test,y_pred) # for r2 score first test then predict
print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(f"R² Score: {r2:.2f}")


# In[68]:


# 55. Consider a dataset CarPrices.csv with features such as horsepower, age, and a target column price.
# Write a Python code to fit a linear regression model to predict price based on the features and
# evaluate the model using MSE, R², and MAE.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

df=pd.read_csv('CarPrices.csv')
x=df[['horsepower','months_old']]
y=df['price']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

ls=LinearRegression()
ls.fit(x_train,y_train)

y_pred= ls.predict(x_test)

r2=r2_score(y_test,y_pred) # for r2 score first test then predict
print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(f"R² Score: {r2:.2f}")


# In[85]:


# 54. Consider a dataset HousePrices.csv with features such as size (in square feet) and a target column
# price (in dollars). Write a Python code to implement Linear Regression using Gradient Descent to
# predict price based on size. Use the complete dataset to train the model. Plot the regression line
# along with the training data points
import pandas as pd
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('HouseData.csv')

x=df[['SqFt']]
y=df['Price']

s=StandardScaler()
x_scaled=s.fit_transform(x)

x_train,x_test,y_train,y_test= train_test_split(x_scaled,y,test_size=0.2,random_state=42)

sgd=SGDRegressor(max_iter=1000,learning_rate='optimal',eta0=0.01,random_state=42)
sgd.fit(x_train,y_train)
y_pred=sgd.predict(x_scaled) # point to be note here x_scaled we have to do all data 

plt.scatter(x, y,color='blue')
plt.plot(x,y_pred,color='red')
plt.show()


# In[97]:


# 53. Consider a dataset CarPrices.csv with columns age, mileage, and price. Write a Python code to fit a
# multiple linear regression model with age and mileage as independent variables and price as the
# dependent variable. Use the complete dataset to train the model. Plot the actual prices against the
# predicted prices and create a residual plot.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('CarPrices.csv')
df=df.dropna()
x=df[['months_old','horsepower']]
y=df['price']



ls=LinearRegression()
ls.fit(x,y)
y_pred= ls.predict(x)


plt.scatter(y,y_pred,color='red')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue') # point to be note 
plt.show()

res=y-y_pred
sns.residplot(x=y_pred,y=res,lowess=True,line_kws={'color':'red'})
plt.show()


# In[104]:


# 52. Consider a dataset HousePrices.csv with columns square_feet and price. Write a Python code to fit
# a polynomial regression model with square_feet as the independent variable and price as the
# dependent variable. Use the complete dataset to train the model. Plot the fitted polynomial model
# along with the trained data points and create a residual plot



# same as 54 + poly is like standrd scaler 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('HousePrices.csv')
df = df.dropna()


X = df[['SqFt']]  # Feature
y = df['Price']  # Target


poly = PolynomialFeatures(degree=2)  
X_poly = poly.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train Polynomial Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on full dataset
y_pred = model.predict(X_poly)

# Plot Data and Polynomial Regression Line
plt.figure(figsize=(10, 5))
plt.scatter(df['SqFt'], y, color='blue', label="Actual Prices", alpha=0.6)
plt.scatter(df['SqFt'], y_pred, color='red', label="Predicted Prices", alpha=0.6)
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Polynomial Regression: House Prices")
plt.legend()
plt.show()

# Residual plot
residuals = y - y_pred
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.axhline(y=0, color='black', linestyle='dashed', linewidth=1)  # Reference line at 0
plt.xlabel("Predicted Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()


# In[111]:


# 51. Consider Salary.csv , with years_of_experience and salary. Write a python code for fitting best fit
# simple linear regression with independent variable years_of_experience and dependent variable
# salary. Use complete dataset to train model. Plot :fitted model along with trained data points ,
# residual plot.

# same as 54 +53+52
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Salary.csv')
df=df.dropna()
print(df)
# Define independent (X) and dependent (y) variables
X = df[['Years of Experience']]  # Independent variable
y = df['Salary']  # Dependent variable

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict salaries
y_pred = model.predict(X)

# Print model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Model Evaluation
print("\nMean Absolute Error:", mean_absolute_error(y, y_pred))
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("R² Score:", r2_score(y, y_pred))

# Plot Best-Fit Line
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', linewidth=2, label="Best-Fit Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression: Salary vs Experience")
plt.legend()
plt.show()

# Residual Plot
residuals = y - y_pred

plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Predicted Salary")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# In[112]:


# 49.
# 2. build a logistic regression model that can predict whether a customer is likely to churn based on
# the features in file named customer_churn.csv.
# customer_id: Unique customer ID, age: Customer age,
# gender: Customer gender (male/female)
# account_length: Length of the customer's account (in months)
# international_plan: Whether the customer has an international plan (yes/no)
# voice_mail_plan: Whether the customer has a voice mail plan (yes/no)
# number_vmail_messages: Number of voice mail messages
# total_day_calls: Total day calls
# total_night_calls: Total night calls
# total_intl_calls: Total international calls
# • churn: Whether the customer churned (yes/no)

from sklearn.linear_model import LinearRegression


# In[123]:


# 48.the service) based on various features such as age, account length, and monthly charges. The
# company has historical data on customers, including whether they churned or not.
# Age: (in years), Account_Length: (in months), Monthly_Charges: (in dollars)
# Churn: Target variable (1 if the customer churned, 0 otherwise)
# 'Age': [25, 34, 45, 29, 50, 38, 42, 35, 48, 55],
# 'Account_Length': [12, 24, 36, 18, 48, 30, 42, 24, 36, 60],
# 'Monthly_Charges': [70, 90, 80, 100, 60, 80, 90, 70, 80, 100],
# 'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
# build a logistic regression model using python that can predict the probability of a customer churning
# by spliting 80% of given data for training. Predict targets on trained model of test data. Print
# accuracy

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

# Corrected Data
data = {
    'Age': [25, 34, 45, 29, 50, 38, 42, 35, 48, 55],  # Corrected age values
    'Account_Length': [112, 24, 36, 18, 48, 30, 42, 24, 36, 60],
    'Monthly_Charges': [70, 90, 80, 100, 60, 80, 90, 70, 80, 100],  # Corrected charges
    'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Corrected churn values
}

# Create DataFrame
df = pd.DataFrame(data)

# Define Features & Target
X = df[['Age', 'Account_Length', 'Monthly_Charges']]
y = df['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)  # Corrected syntax

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# In[129]:


# 47. Use sklearn.datasets.fetch_20newsgroups() dataset (all sets) to classify newsgroup documents into
# their respective categories using a Multinomial Naive Bayes classifier. Load the dataset and explore
# the categories available. Preprocess the text data using CountVectorizer to convert text documents
# into a matrix of token counts. Split the dataset and Create a Multinomial Naive Bayes model. Fit the
# model on the training data and print accuracy.

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

df=fetch_20newsgroups(subset='all')
x=df.data
y=df.target

c=CountVectorizer(stop_words='english')
x_c=c.fit_transform(x)

x_train,x_test,y_train,y_test= train_test_split(x_c,y,test_size=0.2,random_state=42)

m=MultinomialNB()
m.fit(x_train,y_train)
y_pred=m.predict(x_test)

print(accuracy_score(y_test,y_pred))


# In[137]:


# 45. Implement a Gaussian Naive Bayes classifier to predict whether a patient has diabetes based on
# various health metrics of PIMA.csv dataset. The dataset consists of information from 768 female
# Pima Indians aged 21 and older, initially gathered by the National Institute of Diabetes and Digestive
# and Kidney Diseases. Target variable: Diabetes (binary, 0 or 1) Attributes: Pregnancies, OGTT (Oral
# Glucose Tolerance Test), Blood pressure, Skin thickness, Insulin, BMI (Body Mass Index), Age,
# Pedigree diabetes function.Load the Dataset from a CSV file. Provide summary statistics for the
# dataset. Split the data into training and testing sets (80% train, 20% test). Instantiate a Gaussian
# Naive Bayes model and fit it on the training data. Predict diabetes status for the test set. Discuss any
# biases in the dataset and how they may affect model performance
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('PIMA.csv')
x=df.drop(columns=['Outcome'])
y=df['Outcome']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

g=GaussianNB()
g.fit(x_train,y_train)

y_pred=g.predict(x_test)
print(accuracy_score(y_test,y_pred))


# In[138]:


# 44. The breast cancer dataset is available in the sklearn.datasets module and can be loaded using
# load_breast_cancer(). implement a Naive Bayes Classifier on the Breast Cancer Dataset using
# python's sklearn library. Assume that all features of datasets are continuous variables and must be
# used for building model. Perform following task. load data set, Split data for training and testing,
# Build a model and Predict labels of test data.

from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=load_breast_cancer()
x=df.data
y=df.target

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

g=GaussianNB()
g.fit(x_train,y_train)

y_pred=g.predict(x_test)
print(accuracy_score(y_test,y_pred))


# In[145]:


# 40. apply Principal Component Analysis (PCA) to a house price prediction dataset to reduce its dimensionality
# and then use a regression model to predict house prices. You will evaluate the model's performance using
# metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("HousePrices.csv")
df= pd.get_dummies(df,drop_first=True)
x=df.drop(columns=['Price'])
y=df['Price']

s=StandardScaler()
x_s=s.fit_transform(x)

pca= PCA(n_components=2)
x_pca=pca.fit_transform(x_s)

x_train,x_test,y_train,y_test= train_test_split(x_pca,y,test_size=0.2,random_state=42)
ls=LinearRegression()
ls.fit(x_train,y_train)

y_pred=ls.predict(x_test)

r2=r2_score(y_test,y_pred) # for r2 score first test then predict
print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(f"R² Score: {r2:.2f}")




# In[146]:


# 39. develop a linear regression model to predict house prices based on various features. You will use a
# dataset that contains information about houses, including features such as the number of bedrooms,
# square footage, and location. You will evaluate the model's performance using metrics such as Mean
# Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. Show performance of same model
# when PCA reduced data set is used.

# same as 40 just do with and without pca


# In[148]:


# 38. Dataset: You will use the Iris dataset, which consists of 150 samples of iris flowers, each described by 4
# features (sepal length, sepal width, petal length, and petal width). The target variable indicates the
# species of the iris flower (Setosa, Versicolor, or Virginica). apply both Linear Discriminant Analysis (LDA)
# and Principal Component Analysis (PCA) to the Iris dataset. You will reduce the dimensionality of the
# dataset using both techniques and visualize the results in 2D scatter plots. You will then compare the
# effectiveness of LDA and PCA in terms of class separability.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

df=load_iris()
x=df.data
y=df.target

colors=['red','blue','yellow']
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x)

lda=LDA(n_components=2)
x_lda=lda.fit_transform(x,y)

plt.scatter(x_pca[:,0],x_pca[:,1],c=y)
plt.show()
plt.scatter(x_lda[:,0],x_lda[:,1],c=y)
plt.show()


# In[149]:


# 37. Objective: The goal of this practical exam is to apply Linear Discriminant Analysis (LDA) to the Breast
# Cancer dataset using the Scikit-learn library. You will reduce the dimensionality of the dataset from
# multiple features to 1 dimension and visualize the results using Matplotlib.
# • load the Breast Cancer dataset
# • initialize the LDA to reduce the dataset to 1 dimension.
# • Fit the LDA model to the Breast Cancer dataset and transform the dataset to obtain the LDAreduced representation.
# Visualization


# for 1d array 
# plt.scatter(x_lda,np.zeros_like(x_lda))
# plt.yticks([])


# In[150]:


# 36. apply Linear Discriminant Analysis (LDA) to the Iris dataset using the Scikit-learn library. You will
# preprocess the data using label encoding, perform LDA to reduce the dimensionality of the dataset, and
# visualize the results in a 2D scatter plot.


# In[151]:


# 35. Apply Linear Discriminant Analysis (LDA) to the Wine dataset to reduce its dimensionality and classify the
# types of wine based on their chemical properties. You will visualize the LDA-reduced dataset in a 2D
# scatter plot and evaluate the classification performance.
# Dataset: You will use the Wine dataset, which consists of 178 samples of wine, each described by 13
# features representing different chemical properties. The target variable indicates the type of wine, which
# can take on one of three classes (1, 2, or 3).

# remember that random forest classifier 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
X = LDA(n_components=2).fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier().fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.show()


# In[153]:


# 34. Apply Linear Discriminant Analysis (LDA) to the Iris dataset to reduce its dimensionality from 4 to 2
# dimensions. You will visualize the LDA-reduced dataset in a 2D scatter plot using Matplotlib.
# instructions:
# Load the Iris dataset
# Briefly explore the dataset to understand its structure. Print the shape of the dataset and the first few
# rows to get an overview of the features and target variable.
# Initialize the PCA object to reduce the dataset to 2 dimensions.
# Fit the LDA model to the Iris dataset and transform the dataset
# Use Matplotlib to create a scatter plot of the LDA-reduced dataset.
# Color the points based on their respective species (Setosa, Versicolor, Virginica) to visualize how well the
# LDA has separated the different classes.
# Add appropriate titles and labels to the axes of the plot.
# Include a legend to indicate which colors correspond to which species.
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Standardizing the features
X_scaled = StandardScaler().fit_transform(X)

# Apply LDA to reduce to 2 dimensions
X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X_scaled, y)

# Create a scatter plot
# colors = ['red', 'green', 'blue']
# for i, color in enumerate(colors) :
#     plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color=color, )

plt.scatter(X_lda[:,0],X_lda[:,1],c=y)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA: Iris Dataset Projection')
plt.legend()
plt.grid()


# In[154]:


# 33. Apply Principal Component Analysis (PCA) to the Iris dataset to reduce its dimensionality from 4 to 2
# dimensions. You will visualize the PCA-reduced dataset in a 2D scatter plot using Matplotlib.
# instructions:
# Load the Iris dataset
# Briefly explore the dataset to understand its structure. Print the shape of the dataset and the first few
# rows to get an overview of the features and target variable.
# Initialize the PCA object to reduce the dataset to 2 dimensions.
# Fit the PCA model to the Iris dataset and transform the dataset
# Use Matplotlib to create a scatter plot of the PCA-reduced dataset.
# Color the points based on their respective species (Setosa, Versicolor, Virginica) to visualize how well the
# PCA has separated the different classes.
# Add appropriate titles and labels to the axes of the plot.
# Include a legend to indicate which colors correspond to which species.


# In[155]:


# 1. Prepare decision tree classifier in python using sklearn library for data set
# diabetes.csv - (pregnant, glucose, bp, skin, insulin, bmi, pedigree, age, label). Use all features except label
# as independent variable. Use complete dataset for training and limit the depth of tree upto 3 levels and
# plot using
# tree.plotTree()method.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset
df = pd.read_csv("diabetes.csv")


X = df.drop(columns=['label'])  # All features except 'label'
y = df['label']  # Target variable


tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)

tree_model.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"])
plt.title("Decision Tree Classifier (Depth = 3)")
plt.show()


# In[156]:


# 2. Using the diabetes dataset, implement a Decision Tree Classifier and determine the importance of each
# feature in predicting diabetes. Generate different plots to justify your model.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
df = pd.read_csv("diabetes.csv")

# Split data into features (X) and target (y)
X = df.drop(columns=['label'])  # Independent variables
y = df['label']  # Target variable

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model
tree_model.fit(X_train, y_train)

# Make predictions
y_pred = tree_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Plot Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"])
plt.title("Decision Tree Classifier (Depth = 3)")
plt.show()

# Get feature importance
feature_importance = tree_model.feature_importances_

# Plot feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=feature_importance, palette="viridis")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Predicting Diabetes")
plt.xticks(rotation=45)
plt.show()


# In[159]:


# 3. Implement a Decision Tree Classifier and compare its performance with a SVM Classifier on the iris dataset.
# Display the accuracy of both models.
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.svm import SVC


# In[160]:


# 4. Build decision tree classifier for iris data set. One with maximum leaf nodes up to 8 and another one with
# minimum sample per leaf as 5. Compare accuracy of both models.


# In[161]:


# 15. Develop a Python program to implement DBSCAN using the sklearn library. Use a 2D dataset (ex.
# make_moons) for clustering.
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=0.5,min_samples=5)
label= db.fit_predict(x)
plt.scatter(x[:0],x[:,1],c=label)


# In[ ]:




