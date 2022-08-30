#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load in the essentials
import mkl
import pandas as pd
import numpy as np

# Load in visual libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load in the dataset
dataset = pd.read_csv('50_startups.csv')
dataset.head() # Show top 5 rows


# In[3]:


# Create a statistical summary table
stats_summary = dataset.describe()
stats_summary


# In[4]:


# Create a correlation table
correlation_table = dataset.corr()
correlation_table


# In[5]:


# Enable inline plotting
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


# Create a scatter plot
x = dataset['Administration']
y = dataset['Profit']
plt.scatter(x,y)
plt.title("Scatter Plot")
plt.xlabel('Administration')
plt.ylabel('Profit'); # Semicolon takes away unnecessary info


# In[7]:


# Create a bar plot
plt.bar(dataset['State'],dataset['Marketing Spend'])
plt.title("Bar Plot")
plt.xlabel('State')
plt.ylabel('Marketing Spend');


# In[8]:


# Create a box plot
sns.boxplot(y='Profit', data=dataset)
plt.title("Box Plot")
plt.ylabel('Profit');


# In[9]:


# Create a violin plot
sns.violinplot(y='Profit', data=dataset)
plt.title("Violin Plot")
plt.ylabel('Profit');


# In[10]:


# Add an X Variable
sns.violinplot(x='State', y='Profit', data=dataset)
plt.title("Violin Plot")
plt.xlabel('State');
plt.ylabel('Profit');


# In[11]:


# Change the Color
sns.swarmplot(x='State', y='Profit', data=dataset, color ='black')
plt.title("Swarm Plot")
plt.xlabel('State');
plt.ylabel('Profit');


# In[12]:


# Overlaying Plots
sns.violinplot(x='State', y='Profit', data=dataset)
sns.swarmplot(x='State', y='Profit', data=dataset, color ='black')


# In[13]:


# Create a histogram
sns.histplot(dataset['Profit']);


# In[14]:


# Combine plots
sns.jointplot(x='Profit', y='Marketing Spend', data=dataset)


# In[15]:


# Plot multiple pairwise bivariate distributions in a dataset
sns.pairplot(data=dataset)


# In[16]:


# Look at the data types
dataset.dtypes


# In[17]:


# Look for null values and sum the occurences
dataset.isnull().sum()


# In[18]:


# Count the string categories
dataset['State'].value_counts()


# In[19]:


# Encode the string or state columns
dataset = pd.get_dummies(dataset, drop_first=True)


# In[20]:


dataset.head()


# In[21]:


# Load in the ML libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[22]:


# Check the columns
dataset.columns


# In[23]:


# Define our variables (we want to predict Profit)
features = ['R&D Spend', 'Administration', 'Marketing Spend',
       'State_Florida', 'State_New York']
target = ['Profit']


# In[24]:


# X & y (from basic Linear Regression equation)
X = dataset[features]
y = dataset[target]


# In[25]:


# Create a training and test set (test 20%, train 80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[26]:


# Fit the data to our ML model
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)
y_pred


# In[57]:


y_test


# In[27]:


# Check the model performance (the accuracy of the model)
r2_score(y_test, y_pred)


# In[28]:


dataset.head()


# In[29]:


# Add Prediction column at location 4 (after Profit column)
dataset.insert(4, 'Predictions', lin.predict(X))
dataset.head()


# In[30]:


# Save as csv
dataset.to_csv('full_dataset.csv')

