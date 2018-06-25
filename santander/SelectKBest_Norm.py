
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('data/train.csv', index_col=0)
train.head()


# In[30]:


X = train.drop('target', axis=1)
y = train.target


# ### Normalizer

# In[31]:


from sklearn.preprocessing import normalize

X = normalize(X)


# ### SelectKBest

# In[32]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

X = SelectKBest(mutual_info_regression, k=500).fit_transform(X, y)
X.shape


# In[33]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[35]:


def rmsle_metric(y_pred,y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))

#y_pred = y_pred.astype('float64')
#y_test = y_test.values.astype('float64')

rmsle = rmsle_metric(y_pred, y_test)
#msle = mean_squared_log_error(y_test, y_pred)

rmsle


# ### SVM

# In[10]:


from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

print(rmsle_metric(y_pred, y_test))


# ### Results
# 
# 
