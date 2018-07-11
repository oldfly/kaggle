
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


train = pd.read_csv('data/train.csv', index_col=0)
train.head()


# In[20]:


X = train.drop('target', axis=1)
y = train.target

del train


# In[44]:


from scipy.stats import skew, kurtosis

def aggregate_row(row):
    non_zero_values = row.iloc[row.nonzero()].astype(np.float)
    zero_values = (row == 0).astype(np.float)
    agg = {
        'mean': row.mean(),
        'std': row.std(),
        'max': row.max(),
        'min': row.min(),
        'sum': row.sum(),
        'skewness': skew(row),
        'kurtosis': kurtosis(row),
        'median': row.median(),
        'q1': np.percentile(row, q=25),
        'q3': np.percentile(row, q=75),
        'log_mean': np.log1p(row).mean(),
        'log_std': np.log1p(row).std(),
        'log_max': np.log1p(row).max(),
        'log_min': np.log1p(row).min(),
        'log_sum': np.log1p(row).sum(),
        'log_skewness': skew(np.log1p(row)),
        'log_kurtosis': kurtosis(np.log1p(row)),
        'log_median': np.log1p(row).median(),
        'log_q1': np.percentile(np.log1p(row), q=25),
        'log_q3': np.percentile(np.log1p(row), q=75),
        'count': row.count(),
    }
    
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
                        'non_zero_fraction': np.nan,
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
    if zero_values.empty:
        aggregations['zero_count'] = np.nan
        aggregations['zero_fraction'] = np.nan
    else:
        aggregations['zero_count'] = zero_values.sum()
        aggregations['zero_fraction'] = np.float(zero_values.count() / row.count())
    
    aggregations.update(agg)
    return pd.Series(aggregations)


# In[46]:


X_agg = X.apply(aggregate_row, axis=1)
X_agg.head()


# In[49]:


from sklearn.model_selection import train_test_split

X_add = pd.concat([X, X_agg], axis=1)

y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_add, y_log, test_size=0.1, random_state=42
)

print(X_train.shape, X_test.shape)


# In[24]:


def rmsle_metric(y_test, y_pred): 
    assert len(y_test) == len(y_pred)
    #y_test = np.expm1(y_test)
    #y_pred = np.expm1(y_pred)
    rmsle = np.sqrt(np.mean((y_pred - y_test)**2))
    return ('RMSLE', rmsle, False)


# In[50]:


# import lightgbm as lgb

# gbm = lgb.LGBMRegressor(
#     objective='regression',
#     num_leaves=31,
#     learning_rate=0.008,
#     n_estimators=600,
#     zero_as_missing=True
# )

# gbm.fit(X_train, y_train,
#         eval_set=[(X_test, y_test)],
#         eval_metric=rmsle_metric,
#         early_stopping_rounds=100
# )

import lightgbm as lgb
from sklearn.model_selection import ShuffleSplit

kf = ShuffleSplit(n_splits=10, random_state=42)

gbm = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=10,
    learning_rate=0.008,
    n_estimators=800,
    #reg_lambda=2.0,
    #reg_alpha=1.0,
    max_depth=10,
    n_jobs=-1,
    zero_as_missing=True
)

scores = []
for train_index, test_index in kf.split(X_add):
    X_train, X_test = X_add.values[train_index], X_add.values[test_index]
    y_train, y_test = y_log[train_index], y_log[test_index]
    gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=rmsle_metric,
        early_stopping_rounds=100,
        verbose=False
    )
    scores.append(gbm.best_score_['valid_0']['RMSLE'])


# In[51]:


print(np.mean(scores))
print(np.std(scores))


# In[ ]:


from sklearn.externals import joblib

joblib.dump(gbm, 'LightGBM-Raw-Agg.pkl')


# In[ ]:


from sklearn.externals import joblib

gbm = joblib.load('LightGBM-Raw-Agg.pkl')


# In[11]:


test = pd.read_csv('data/test.csv', index_col=0)
test.head()


# In[12]:


ids = test.reset_index()['ID']


# In[13]:


ids[0]


# In[14]:


import gc

try:
    del X, X_train, X_test
except:
    pass

X_agg = test.apply(aggregate_row, axis=1)

X_add = pd.concat([test, X_agg], axis=1)

del test, X_agg
gc.collect()

y_pred = gbm.predict(X_add)


# In[15]:


y_pred[0]


# In[16]:


y_pred = np.expm1(y_pred)


# In[17]:


pd.DataFrame(data=y_pred, index=ids, columns=['target']).to_csv('data/submit_0710_2.csv')

