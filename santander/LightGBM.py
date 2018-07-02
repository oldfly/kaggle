
# coding: utf-8

# In[9]:


import lightgbm as lgb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


train = pd.read_csv('data/train.csv', index_col=0)

X = train.drop('target', axis=1)
y = train.target


# In[55]:


from sklearn.decomposition import PCA

pca = PCA(
    copy=True, iterated_power=7, n_components=100, 
    random_state=None, svd_solver='auto', tol=0.0, whiten=False
)
X_pca = pca.fit_transform(X)


# In[60]:


from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
X_pca = minmax.fit_transform(X)
y_log = np.log1p(y)


# In[61]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_log, test_size=0.2, random_state=42
)


# In[62]:


def rmsle_metric(y_test, y_pred) : 
    assert len(y_test) == len(y_pred)
    y_test = np.exp(y_test)-1
    y_pred = np.exp(y_pred)-1
    rmsle = np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
    return ('RMSLE', rmsle, False)


# In[63]:


gbm = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=31,
    learning_rate=0.01,
    n_estimators=1000
)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=rmsle_metric,
        early_stopping_rounds=100
)


# In[64]:


y_pred = gbm.predict(X_test)
print(rmsle_metric(y_test, y_pred))


# In[66]:


from sklearn.externals import joblib

joblib.dump(gbm, 'LightGBM_log_y_1_427.pkl')

