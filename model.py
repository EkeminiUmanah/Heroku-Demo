#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[2]:


dataset = pd.read_csv('hiring.csv')


# In[3]:


dataset.head()


# In[5]:


#dataset.info()


# In[6]:


dataset['experience'].fillna(0, inplace = True)


# In[7]:


dataset['test_score'].fillna(dataset['test_score'].mean(), inplace = True)


# In[8]:


X = dataset.iloc[:, :3]


# In[10]:


#Convert words in 'expereience' column to integers
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve' :12, 'zero':0, 0:0}
    return word_dict[word]


# In[11]:


X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))


# In[12]:


y = dataset.iloc[:, -1]


# In[13]:


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.


# In[14]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[15]:


#fitting model with training data
regressor.fit(X,y)


# In[18]:


#Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))


# In[19]:


#loading model to compare results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,9,6]]))


# In[ ]:




