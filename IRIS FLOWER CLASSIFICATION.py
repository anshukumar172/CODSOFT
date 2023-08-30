#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[6]:


df = pd.read_csv(r"D:\CODSOFT\iris.csv")
df


# In[7]:


# to display stats about data
df.describe()


# In[8]:


# to basic info about datatype
df.info()


# In[9]:


# to display no. of samples on each class
df['species'].value_counts()


# In[10]:


# check for null values
df.isnull().sum()


# In[11]:


df.keys()


# In[12]:


df.shape


# In[13]:


df.plot()


# In[14]:


df.max()


# In[15]:


sns.boxplot(x="species", y='petal_length', data=df ) 
plt.show()


# In[16]:


sns.boxplot(x="species", y="petal_width", data=df)


# In[17]:


sns.pairplot(df,hue = "species")


# In[18]:


df['sepal_length'].hist()
plt.title('Histogram')


# In[19]:


df['sepal_width'].hist()
plt.title('Histogram')


# In[20]:


df['petal_length'].hist()
plt.title('Histogram')


# In[21]:


df['petal_width'].hist()
plt.title('Histogram')


# In[22]:


df['species'].hist()
plt.title('Histogram')


# In[23]:


# scatterplot
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['sepal_width'], c = colors[i], label=species[i])
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.legend()


# In[24]:


colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['petal_length'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.legend()


# In[25]:


colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['petal_length'], c = colors[i], label=species[i])
plt.xlabel("sepal_length")
plt.ylabel("petal_length")
plt.legend()


# In[26]:


colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_width'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("sepal_width")
plt.ylabel("petal_width")
plt.legend()


# In[27]:


df.corr()


# In[28]:


corr = df.corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap = 'plasma')


# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[30]:


df['species'] = le.fit_transform(df['species'])
df.head()


# In[31]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['species'])
Y =df['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[32]:


#logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[33]:


# model training
model.fit(x_train, y_train)


# In[34]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[35]:


y_pred=model.predict(x_train)
y_pred


# In[36]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)


# In[37]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[38]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)


# In[39]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[40]:


#Dataset source:https://www.kaggle.com/datasets/arshid/iris-flower-dataset


# This project is made by Anshu kumar during internship at codsoft

# In[ ]:




