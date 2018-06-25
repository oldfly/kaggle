
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


# In[3]:


X = train.drop('target', axis=1)
y = train.target


# In[4]:


from sklearn.decomposition import PCA

pca = PCA(n_components=500)
X_pca = pca.fit_transform(X)
X_pca.shape


# In[36]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)


# In[37]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[70]:


def rmsle_metric(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))

#y_pred = y_pred.astype('float64')
#y_test = y_test.values.astype('float64')

rmsle = rmsle_metric(y_pred, y_test)
#msle = mean_squared_log_error(y_test, y_pred)


# In[71]:


rmsle


# A aplicação de SelectKBest apresentou melhor performance no teste com regressão linear com RMSLE de 1,942.
# 
# Vamos fazer mais um teste com SVM.

# ### SVM

# In[72]:


from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

print(rmsle_metric(y_pred, y_test))


# A Aplicação de SelecKBest performou melhor no teste com SVM com RMSLE de 1.7021190276421692 (2.869542599093222e-07 menor). Vamos seguir aquela linha de experimentos e abandonar esta.
