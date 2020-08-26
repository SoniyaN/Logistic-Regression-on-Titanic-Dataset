#!/usr/bin/env python
# coding: utf-8

# In[196]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[197]:


df = pd.read_csv('C:/Users/Hp/Downloads/titanic.csv')
df.head()


# In[198]:


df.isnull().sum()


# In[199]:


del df['deck'] # Contains 688 unknown values
del df['embark_town'] # Replaced with 'embarked'
df.age = df.age.fillna(df.age.mean())
df.embarked = df.embarked.fillna(df.embarked.mode()[0])


# In[200]:


df.isnull().sum()


# In[201]:


df.describe()


# # DATA ANALYSIS

# In[202]:


a=['age','fare']
b=['survived','pclass','sex','parch','embarked','who','adult_male','alive','alone','sibsp','class']


# In[203]:


e_count=df['embarked'].value_counts()
#d_count=df['deck'].value_counts()
c_count=df['class'].value_counts()
p_count=df['pclass'].value_counts()
print(e_count.sort_index())
#print(d_count.sort_index()) 
print(c_count.sort_index()) 
print(p_count.sort_index())


# In[204]:


for i in b:
    sns.countplot(df[i])
    plt.show()


# In[205]:


for i in a:
    sns.swarmplot(y=df[i],x=df['survived'])
    plt.show()


# In[206]:


sns.countplot(df['embarked'],hue=df['survived'])


# In[207]:


sns.countplot(df['class'],hue=df['survived'])


# In[208]:


x = df['survived']
z = df['pclass']
sns.lineplot(x, z, color = 'blue')


# In[209]:


sns.barplot(y=df['survived'],x=df['who'],hue=df['class'])


# In[210]:


sns.lineplot(y=df['survived'],x=df['who'],hue=df['embarked'])


# # ANALYSIS REPORT
1. Variable deck has 688 unknown values.
2. Maximum passengers travelled in Third Class.
3. Most of the passengers had no sibblings and were travelling alone.
4. First class passengers have survived more in number.
5. Survival rate for man is lower compared to woman and child.
# In[211]:


df=pd.get_dummies(df)


# In[212]:


df.head()


# In[213]:


del df['sex_male']
del df['alive_no']
del df['embarked_S']
del df['pclass']
del df['class_Third']
del df['who_child']


# In[214]:


df.info()


# In[226]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,r2_score


# In[227]:


df['adult_male'] = df['adult_male']*1
df['alone'] = df['alone']*1
df.head()


# In[228]:


x = df.drop('survived',axis=1)
y= df['survived']


# In[229]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2) # 20% testing dataset


# In[230]:


xtrain.shape,ytrain.shape


# In[231]:


xtest.shape,ytest.shape


# In[232]:


model = LogisticRegression()


# In[233]:


model.fit(xtrain,ytrain)


# In[234]:


ypred = model.predict(xtest)


# In[235]:


print('accuracy = ',accuracy_score(ytest,ypred))


# In[236]:


print('accuracy = ',r2_score(ytest,ypred))


# In[ ]:




