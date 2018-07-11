
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


train = pd.read_csv('data/train.csv', index_col=0)
train.head()


# In[9]:


X = train.drop('target', axis=1)
y = train.target
del train


# In[10]:


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


# In[11]:


X_agg = X.apply(aggregate_row, axis=1)


# In[12]:


X_agg.head()


# In[14]:


from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=50, random_state=42)
X_fa = fa.fit_transform(X)


# In[15]:


from sklearn.random_projection import GaussianRandomProjection

grp = GaussianRandomProjection(n_components=50, random_state=42, eps=0.1)
X_grp = grp.fit_transform(X)


# In[16]:


# from sklearn.decomposition import PCA

# pca = PCA(n_components=100, random_state=42)
# X_pca = pca.fit_transform(X)


# In[17]:


# from sklearn.decomposition import FastICA

# ica = FastICA(n_components=15, random_state=42)
# X_ica = ica.fit_transform(X)


# In[30]:


from sklearn.model_selection import train_test_split

X_added = pd.concat([
    pd.DataFrame(X_fa),
    #pd.DataFrame(X_srp),
    pd.DataFrame(X_grp),
    #pd.DataFrame(X_pca),
    #pd.DataFrame(X_ica),
    #X.reset_index().drop('ID', axis=1),
    X_agg.reset_index().drop('ID', axis=1)
], axis=1)

y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_added, y_log, test_size=0.1, random_state=42
)

#X_val, X_test, y_val, y_test = train_test_split(
#    X_test, y_test, test_size=0.5, random_state=42
#)

print(X_added.shape)


# In[28]:


from sklearn.metrics import make_scorer

def rmsle_metric(y_test, y_pred) : 
    assert len(y_test) == len(y_pred)
    y_test = np.exp(y_test)-1
    y_pred = np.exp(y_pred)-1
    rmsle = np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
    return ('RMSLE', rmsle, False)

def rmsle_score(y_test, y_pred):
    return rmsle_metric(y_test, y_pred)[1]

grid_scorer = make_scorer(rmsle_score, greater_is_better=False)


# In[26]:


import lightgbm as lgb
from sklearn.model_selection import ShuffleSplit

kf = ShuffleSplit(n_splits=5, random_state=42)

gbm = lgb.LGBMRegressor(
    objective='regression',
    #num_leaves=11,
    learning_rate=0.008,
    n_estimators=1000,
    #reg_lambda=2.0,
    #reg_alpha=1.0,
    max_depth=15,
    n_jobs=-1,
)

scores = []
for train_index, test_index in kf.split(X_added):
    X_train, X_test = X_added.values[train_index], X_added.values[test_index]
    y_train, y_test = y_log[train_index], y_log[test_index]
    gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=rmsle_metric,
        early_stopping_rounds=100,
        verbose=False
    )
    scores.append(gbm.best_score_['valid_0']['RMSLE'])


# In[32]:


import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

params = {
    'num_leaves': [5, 10, 30, 50],
    'reg_lambda': [0., .5, 1., 3],
    'reg_alpha': [0., .5, 1., 3.],
    'max_depth': [5, 10, 20]
}

gbm = lgb.LGBMRegressor(
    objective='regression',
    learning_rate=0.008,
    n_estimators=1000,
    n_jobs=-1
)

grid = GridSearchCV(gbm, params, cv=5, scoring=grid_scorer, verbose=2)

grid.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=rmsle_metric,
    early_stopping_rounds=100,
    verbose=False
)


# outliers + fa + grp

# In[15]:


print(np.mean(scores))
print(np.std(scores))


# In[36]:


y_pred = grid.best_estimator_.predict(X_test)
rmsle_score(y_test, y_pred)


# In[39]:


results = pd.DataFrame(grid.cv_results_)
results.sort_values('rank_test_score').head()


# In[40]:


test = pd.read_csv('data/test.csv', index_col=0)
test.head()


# In[41]:


ids = test.reset_index()['ID']


# In[42]:


from sklearn.decomposition import FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

del X_test, X_train, y_test, y_train

X_agg = test.apply(aggregate_row, axis=1)
X_fa = fa.transform(test)
#X_srp = srp.transform(test)
X_grp = grp.transform(test)
#X_ica = ica.transform(test)
#X_pca = pca.transform(test)


del test

X_added = pd.concat([
    pd.DataFrame(X_fa),
    #pd.DataFrame(X_srp),
    pd.DataFrame(X_grp),
    #pd.DataFrame(X_pca),
    #pd.DataFrame(X_ica),
    X_agg.reset_index().drop('ID', axis=1)
], axis=1)

y_pred = grid.best_estimator_.predict(X_added)
y_pred


# In[43]:


y_pred = np.exp(y_pred) - 1


# In[44]:


y_pred[0]


# In[45]:


ids[0]


# In[46]:


pd.DataFrame(y_pred, index=ids, columns=['target']).to_csv('data/submit_0907.csv')


# In[47]:


pd.DataFrame(y_pred, index=ids, columns=['target']).head()


# In[ ]:


from sklearn.externals import joblib

joblib.dump(grid.best_estimator_, 'GridSearch_LGB')

