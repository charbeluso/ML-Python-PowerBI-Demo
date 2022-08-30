#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load in essentials
import pandas as pd
import numpy as np


# In[3]:


# Load in the dataset
dataset = pd.read_csv('avocado.csv')
dataset.head() # Show top 5 rows


# In[4]:


# Create a statistical summary table
dataset.describe()


# In[5]:


# Check the columns/column names
dataset.columns


# In[6]:


# Check the data types and data structures
dataset.info()


# In[7]:


# Access a single column
dataset['Unnamed: 0']


# In[8]:


# Drop a single column
dataset.drop('Unnamed: 0', axis=1, inplace=True)


# In[9]:


# Check at the dataset
dataset.head()


# In[10]:


# Save a statistical summary table
stats_summary = dataset.describe()
stats_summary


# In[12]:


# Save a correlation table
correlation_table = dataset.corr()
correlation_table

