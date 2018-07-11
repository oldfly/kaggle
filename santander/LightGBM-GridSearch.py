
# coding: utf-8

# ### MinMaxScaling X and y

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
y_log = np.log1p(y)


# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.1, random_state=42
)


# In[4]:


from sklearn.metrics import make_scorer

def rmsle_metric(y_test, y_pred) : 
    assert len(y_test) == len(y_pred)
    y_test = np.exp(y_test)-1
    y_pred = np.exp(y_pred)-1
    rmsle = np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
    return ('RMSLE', rmsle, False)

grid_scorer = make_scorer(lambda y_test, y_pred: rmsle_metric(y_test, y_pred)[1], greater_is_better=False)


# In[8]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, mutual_info_regression


pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('regressor', lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.01,
        silent=False
    ))
])

N_FEATURES_OPTIONS = [50, 100, 300]

param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'regressor__boosting_type': ['gbdt', 'dart'], #'goss', 'rf'],
        'regressor__n_estimators': [50, 100, 500]
    },
    {
        'reduce_dim': [SelectKBest(mutual_info_regression)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'regressor__boosting_type': ['gbdt', 'dart'], #'goss', 'rf'],
        'regressor__n_estimators': [50, 100, 500]
    },
]
reducer_labels = ['PCA', 'NMF', 'KBest']

grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid) #, scoring=grid_scorer)
grid.fit(X_train, y_train) #, **{
    #'regressor__eval_set':[(X_test, y_test)],
    #'regressor__eval_metric':rmsle_metric,
    #'regressor__early_stopping_rounds':100
    #}
#)


# In[9]:


y_pred = grid.predict(X_test)
rmsle_metric(y_test, y_pred)


# In[55]:


from sklearn.externals import joblib

joblib.dump(grid.best_estimator_, 'LightGBM-GridSearch-1_420.pkl')


# In[11]:


grid.cv_results_.keys()


# In[13]:


keys = ['param_reduce_dim', 'param_reduce_dim__n_components', 'param_regressor__boosting_type', 'param_regressor__n_estimators', 'param_reduce_dim__k', 'params',
       'mean_test_score', 'std_test_score', 'rank_test_score','mean_train_score', 'std_train_score']

results = pd.DataFrame(grid.cv_results_)[keys]


# In[17]:


results.sort_values('mean_test_score')


# In[19]:


grid.best_index_


# In[20]:


grid.best_score_


# In[51]:


results[results['rank_test_score']==3]['params'].values[0]


# In[52]:


del pipe2
pipe2 = Pipeline([
    ('reduce_dim', SelectKBest(k=50)),
    ('regressor', lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.01,
        n_estimators=500,
        boosting_type='gbdt'
    ))
])
pipe2.fit(X_train, y_train)


# In[53]:


y_pred = pipe2.predict(X_test)
rmsle_metric(y_test, y_pred)

