
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


# In[10]:


from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=50, random_state=42)
X_fa = fa.fit_transform(X)


# In[11]:


from sklearn.random_projection import SparseRandomProjection

srp = SparseRandomProjection(n_components=50, random_state=42)
X_srp = srp.fit_transform(X)


# In[12]:


from sklearn.random_projection import GaussianRandomProjection

grp = GaussianRandomProjection(n_components=50, random_state=42)
X_grp = grp.fit_transform(X)


# In[17]:


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

# X_val, X_test, y_val, y_test = train_test_split(
#     X_test, y_test, test_size=0.5, random_state=42
# )

print(X_train.shape, X_test.shape)


# In[20]:


from sklearn.metrics.scorer import make_scorer

def rmsle_metric(y_test, y_pred): 
    assert len(y_test) == len(y_pred)
    y_test = np.exp(y_test)-1
    y_pred = np.exp(y_pred)-1
    rmsle = np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
    return rmsle

rmsle_scorer = make_scorer(rmsle_metric, greater_is_better=False)


# In[21]:


from tpot import TPOTRegressor

tpot_config = {
    'xgboost.XGBRegressor': {
        'booster': ['gbtree', 'gblinear', 'dart'],
        'n_estimators': [100, 150],
        'max_depth': range(3, 15),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1]
    },
    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': range(1, 51),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },
    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
}

model = TPOTRegressor(
    generations=20,
    population_size=100,
    n_jobs=2,
    verbosity=2,
    cv=5,
    early_stop=5,
    config_dict=tpot_config,
    scoring=rmsle_scorer
)
model.fit(X_train, y_train.values)


# In[22]:


y_pred_t = model.predict(X_train)
print(rmsle_metric(y_train, y_pred_t))

y_pred = model.predict(X_test)
print(rmsle_metric(y_test, y_pred))

# y_pred_v = gbm.predict(X_val)
# print(rmsle_metric(y_val, y_pred_v))


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

