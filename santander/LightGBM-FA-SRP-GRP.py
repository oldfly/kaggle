
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


train = pd.read_csv('data/train.csv', index_col=0)
train.head()


# In[37]:


X = train.drop('target', axis=1)
y = train.target
del train


# In[38]:


from scipy.stats import skew, kurtosis

def aggregate_row(row):
    non_zero_values = row.iloc[row.nonzero()].astype(np.float)
    if non_zero_values.empty:
        aggregations = {'non_zero_mean': np.nan,
                        'non_zero_std': np.nan,
                        'non_zero_max': np.nan,
                        'non_zero_min': np.nan,
                        'non_zero_sum': np.nan,
                        'non_zero_skewness': np.nan,
                        'non_zero_kurtosis': np.nan,
                        'non_zero_median': np.nan,
                        'non_zero_q1': np.nan,
                        'non_zero_q3': np.nan,
                        'non_zero_log_mean': np.nan,
                        'non_zero_log_std': np.nan,
                        'non_zero_log_max': np.nan,
                        'non_zero_log_min': np.nan,
                        'non_zero_log_sum': np.nan,
                        'non_zero_log_skewness': np.nan,
                        'non_zero_log_kurtosis': np.nan,
                        'non_zero_log_median': np.nan,
                        'non_zero_log_q1': np.nan,
                        'non_zero_log_q3': np.nan,
                        'non_zero_count': np.nan,
                        'non_zero_fraction': np.nan
                        }
    else:
        aggregations = {'non_zero_mean': non_zero_values.mean(),
                        'non_zero_std': non_zero_values.std(),
                        'non_zero_max': non_zero_values.max(),
                        'non_zero_min': non_zero_values.min(),
                        'non_zero_sum': non_zero_values.sum(),
                        'non_zero_skewness': skew(non_zero_values),
                        'non_zero_kurtosis': kurtosis(non_zero_values),
                        'non_zero_median': non_zero_values.median(),
                        'non_zero_q1': np.percentile(non_zero_values, q=25),
                        'non_zero_q3': np.percentile(non_zero_values, q=75),
                        'non_zero_log_mean': np.log1p(non_zero_values).mean(),
                        'non_zero_log_std': np.log1p(non_zero_values).std(),
                        'non_zero_log_max': np.log1p(non_zero_values).max(),
                        'non_zero_log_min': np.log1p(non_zero_values).min(),
                        'non_zero_log_sum': np.log1p(non_zero_values).sum(),
                        'non_zero_log_skewness': skew(np.log1p(non_zero_values)),
                        'non_zero_log_kurtosis': kurtosis(np.log1p(non_zero_values)),
                        'non_zero_log_median': np.log1p(non_zero_values).median(),
                        'non_zero_log_q1': np.percentile(np.log1p(non_zero_values), q=25),
                        'non_zero_log_q3': np.percentile(np.log1p(non_zero_values), q=75),
                        'non_zero_count': non_zero_values.count(),
                        'non_zero_fraction': non_zero_values.count() / row.count()
                        }
    return pd.Series(aggregations)


# In[39]:


X_agg = X.apply(aggregate_row, axis=1)


# In[40]:


X_agg.head()


# In[41]:


from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=50, random_state=42)
X_fa = fa.fit_transform(X)


# In[42]:


from sklearn.random_projection import SparseRandomProjection

srp = SparseRandomProjection(n_components=50, random_state=42)
X_srp = srp.fit_transform(X)


# In[43]:


from sklearn.random_projection import GaussianRandomProjection

grp = GaussianRandomProjection(n_components=50, random_state=42, eps=0.1)
X_grp = grp.fit_transform(X)


# In[60]:


from sklearn.decomposition import PCA

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)


# In[61]:


from sklearn.decomposition import FastICA

ica = FastICA(n_components=15, random_state=42)
X_ica = ica.fit_transform(X)


# In[62]:


from sklearn.model_selection import train_test_split

X_added = pd.concat([
    pd.DataFrame(X_fa),
    pd.DataFrame(X_srp),
    pd.DataFrame(X_grp),
    pd.DataFrame(X_pca),
    pd.DataFrame(X_ica),
    X_agg.reset_index().drop('ID', axis=1)
], axis=1)

y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_added, y_log, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

print(X_train.shape, X_val.shape, X_test.shape)


# In[63]:


def rmsle_metric(y_test, y_pred): 
    assert len(y_test) == len(y_pred)
    y_test = np.exp(y_test)-1
    y_pred = np.exp(y_pred)-1
    rmsle = np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
    return ('RMSLE', rmsle, False)


# In[66]:


import lightgbm as lgb

gbm = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=11,
    learning_rate=0.008,
    n_estimators=1000,
    reg_lambda=2.0,
    reg_alpha=1.0,
    max_depth=5,
    n_jobs=2
)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=rmsle_metric,
        early_stopping_rounds=100
)


# In[67]:


y_pred_t = gbm.predict(X_train)
print(rmsle_metric(y_train, y_pred_t))

y_pred = gbm.predict(X_test)
print(rmsle_metric(y_test, y_pred))

y_pred_v = gbm.predict(X_val)
print(rmsle_metric(y_val, y_pred_v))


# best results until now

# In[27]:


y_pred_t = gbm.predict(X_train)
print(rmsle_metric(y_train, y_pred_t))

y_pred = gbm.predict(X_test)
print(rmsle_metric(y_test, y_pred))

y_pred_v = gbm.predict(X_val)
print(rmsle_metric(y_val, y_pred_v))


# In[69]:


test = pd.read_csv('data/test.csv', index_col=0)
test.head()


# In[70]:


ids = test.reset_index()['ID']


# In[71]:


from sklearn.decomposition import FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

del X_test, X_train, y_test, y_train

X_agg = test.apply(aggregate_row, axis=1)
X_fa = fa.transform(test)
X_srp = srp.transform(test)
X_grp = grp.transform(test)
X_ica = ica.transform(test)
X_pca = pca.transform(test)

del test

X_added = pd.concat([
    pd.DataFrame(X_fa),
    pd.DataFrame(X_srp),
    pd.DataFrame(X_grp),
    pd.DataFrame(X_pca),
    pd.DataFrame(X_ica),
    X_agg.reset_index().drop('ID', axis=1)
], axis=1)

y_pred = gbm.predict(X_added)
y_pred


# In[72]:


y_pred = np.exp(y_pred) - 1


# In[73]:


y_pred[0]


# In[74]:


ids[0]


# In[75]:


pd.DataFrame(y_pred, index=ids, columns=['target']).to_csv('data/submit.csv')

