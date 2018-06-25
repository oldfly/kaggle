
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('data/train.csv', index_col=0)
train.head()


# In[4]:


train.describe()


# ### There are missing values?

# In[5]:


train.isnull().values.any()


# ### Any column is all zeros?

# In[6]:


total_zero_cols = 0
for col, vals in train.iteritems():
    if vals.sum() == 0:
        total_zero_cols += 1

print(total_zero_cols)


# ### Features Correlation

# In[7]:


corr = train.corr()


# In[8]:


corr_matrix = corr.abs()
high_corr_var=np.where(corr_matrix>0.7)
high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]


# In[9]:


to_delete = list(dict(high_corr_var).values())
train = train.drop(to_delete, axis=1)


# ### SelectKBest Features

# In[10]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

X = train.drop('target', axis=1)
y = train.target

X = SelectKBest(mutual_info_regression, k=2000).fit_transform(X, y)
X.shape


# ### PCA

# In[11]:


from sklearn.decomposition import PCA

pca = PCA(n_components=500)
X_pca = pca.fit_transform(X)
X_pca.shape


# ### Train Test Split

# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)


# ### Checkpoint

# In[62]:


import pickle

data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "X": X
}

with open('data/prepared_data.pkl', 'wb') as f:
    pickle.dump(data, f)


# ### Linear Regression Test

# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[14]:


def rmsle_metric(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))

#y_pred = y_pred.astype('float64')
#y_test = y_test.values.astype('float64')

rmsle = rmsle_metric(y_pred, y_test)
#msle = mean_squared_log_error(y_test, y_pred)

rmsle


# A aplicação de SelectKBest e PCA de formas isoladas apresentaram melhor performance no teste com regressão linear com RMSLE de 1,94 ambos.
# 
# Esta linha de experimentos será abandonada

# In[6]:


X = train.drop('target', axis=1)
y = train.target


# In[ ]:


from tpot import TPOTRegressor
from datacleaner import autoclean

model = TPOTRegressor(
    generations=2,
    population_size=50,
    scoring='mean_squared_error',
    n_jobs=4,
    verbosity=2
)
model.fit(autoclean(X), y)

