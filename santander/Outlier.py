
# coding: utf-8

# In[1]:


import lightgbm as lgb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('data/train.csv', index_col=0)

X = train.drop('target', axis=1)
y = train.target


# In[3]:


train.describe()


# In[5]:


X.shape


# In[11]:


X[np.abs(X - X.mean()) <= (3*X.std())].describe()


# In[13]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)


# In[21]:


sns.pointplot(x=1, y=0, data=pd.DataFrame(X_pca))

