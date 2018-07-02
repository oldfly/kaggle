
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('data/train.csv', index_col=0)

X = train.drop('target', axis=1)
y = train.target

### PCA

from sklearn.decomposition import PCA

pca = PCA(
    copy=True, iterated_power=7, n_components=100, 
    random_state=None, svd_solver='auto', tol=0.0, whiten=False
)
X = pca.fit_transform(X)

### Scaler

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
X = minmax.fit_transform(X)
y = np.log1p(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def rmsle_metric(y_test, y_pred) : 
    assert len(y_test) == len(y_pred)
    y_test = np.exp(y_test)-1
    y_pred = np.exp(y_pred)-1
    rmsle = np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


# In[4]:


from tpot import TPOTRegressor


model = TPOTRegressor(
    generations=10,
    population_size=100,
    n_jobs=4,
    verbosity=2,
    cv=3,
    early_stop=3
)
model.fit(X_train, y_train.values)


# In[5]:


def rmsle_metric(y_test, y_pred) : 
    assert len(y_test) == len(y_pred)
    y_test = np.exp(y_test)-1
    y_pred = np.exp(y_pred)-1
    rmsle = np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
    return rmsle

y_pred = model.predict(X_test)
print(rmsle_metric(y_test, y_pred))


# In[6]:


from sklearn.externals import joblib

joblib.dump(model.fitted_pipeline_, 'PCA_y_log_TPOT_1_475.pkl')

