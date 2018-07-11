
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
del train


# In[4]:


from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=100, random_state=42)
X_fa = fa.fit_transform(X)


# In[5]:


from sklearn.random_projection import SparseRandomProjection

srp = SparseRandomProjection(n_components=100, random_state=42)
X_srp = srp.fit_transform(X)


# In[6]:


from sklearn.random_projection import GaussianRandomProjection

grp = GaussianRandomProjection(n_components=100, random_state=42)
X_grp = grp.fit_transform(X)


# In[7]:


from sklearn.model_selection import train_test_split

X_added = pd.concat([
    pd.DataFrame(X_fa),
    pd.DataFrame(X_srp),
    pd.DataFrame(X_grp),
], axis=1)

y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_added, y_log, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

print(X_train.shape, X_val.shape, X_test.shape)


# In[8]:


def rmsle_metric(y_test, y_pred): 
    assert len(y_test) == len(y_pred)
    y_test = np.exp(y_test)-1
    y_pred = np.exp(y_pred)-1
    rmsle = np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
    return ('RMSLE', rmsle, False)


# In[19]:


import lightgbm as lgb

gbm = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=11,
    learning_rate=0.008,
    n_estimators=1000,
    reg_lambda=2.0,
    max_depth=5,
)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=rmsle_metric,
        early_stopping_rounds=100
)


# In[20]:


y_pred_t = gbm.predict(X_train)
print(rmsle_metric(y_train, y_pred_t))

y_pred = gbm.predict(X_test)
print(rmsle_metric(y_test, y_pred))

y_pred_v = gbm.predict(X_val)
print(rmsle_metric(y_val, y_pred_v))


# In[43]:


test = pd.read_csv('data/test.csv', index_col=0)
test.head()


# In[44]:


ids = test.reset_index()['ID']


# In[45]:


from sklearn.decomposition import FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

X_fa = fa.transform(test)

X_srp = srp.transform(test)

X_grp = grp.transform(test)

X_added = pd.concat([
    pd.DataFrame(X_fa),
    pd.DataFrame(X_srp),
    pd.DataFrame(X_grp),
], axis=1)

y_pred = gbm.predict(X_added)
y_pred


# In[46]:


y_pred = np.exp(y_pred) - 1


# In[47]:


y_pred[0]


# In[48]:


ids[0]


# In[50]:


pd.DataFrame(y_pred, index=ids, columns=['target']).to_csv('data/submit.csv')

