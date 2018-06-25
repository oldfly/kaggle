
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


# ### Scaler

# In[4]:


from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
X = minmax.fit_transform(X)

pd.DataFrame(X).describe()


# ### SelectKBest

# In[5]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

X = SelectKBest(mutual_info_regression, k=500).fit_transform(X, y)
X.shape


# In[10]:


pd.DataFrame(X).describe()


# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[30]:


def rmsle_metric(y_pred,y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


# ### Linear Regression

# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

params = {
    'normalize': [True, False],
    'fit_intercept': [True, False]
}

lr = LinearRegression()
lr_gs = GridSearchCV(lr, params)
lr_gs.fit(X_train, y_train)

y_pred = lr_gs.predict(X_test)

print(rmsle_metric(y_pred, y_test))

lr_gs.best_estimator_


# ### SVM

# In[25]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

params = {
    'C': [.5, 1.5],
    'epsilon': [0.05, 0.1, 0.3],
    'kernel': ['poly', 'rbf'],
}

svr = SVR()
svr_gs = GridSearchCV(svr, params)
svr_gs.fit(X_train, y_train)

y_pred = svr_gs.predict(X_test)

print(rmsle_metric(y_pred, y_test))

svr_gs.best_estimator_


# ### KNN

# In[19]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

params = {
    'n_neighbors': [5, 15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree']
}

knn = KNeighborsRegressor()
knn_gs = GridSearchCV(knn, params, n_jobs=2)
knn_gs.fit(X_train, y_train)

y_pred = knn_gs.predict(X_test)

print(rmsle_metric(y_pred, y_test))


# In[22]:


knn_gs.best_estimator_


# ### ExtraTreeRegressor

# In[32]:


from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import GridSearchCV

params = {
    'criterion': ['mse', 'mae'],
    'splitter': ['best', 'random'],
    'max_depth': [50, 100, 500]
}

tree = ExtraTreeRegressor()
tree_gs = GridSearchCV(tree, params, n_jobs=2)
tree_gs.fit(X_train, y_train)

y_pred = tree_gs.predict(X_test)

print(rmsle_metric(y_pred, y_test))


# In[33]:


tree_gs.best_estimator_


# ### Neural Network

# In[52]:


from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

params = {
    'hidden_layer_sizes': [(200, 100), (300, 100)],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'max_iter': [800],
    'learning_rate': ['adaptive']
}

start = datetime.now()

nn = MLPRegressor()
nn_gs = GridSearchCV(nn, params)
nn_gs.fit(X_train, y_train)

y_pred = nn_gs.predict(X_test)

print(datetime.now() - start)
print(rmsle_metric(y_pred, y_test))

nn_gs.best_estimator_


# In[48]:


pd.DataFrame(nn_gs.cv_results_).sort_values('rank_test_score').head()[
    ['param_activation', 'param_hidden_layer_sizes', 'param_solver', 'rank_test_score']
]


# In[47]:


nn_gs.cv_results_


# ### Stacking

# In[41]:


from mlxtend.regressor import StackingRegressor

stregr = StackingRegressor(
    regressors=[
        tree_gs.best_estimator_,
        knn_gs.best_estimator_,
        lr_gs.best_estimator_,
        svr_gs.best_estimator_
    ], 
    meta_regressor=svr_gs.best_estimator_
)

stregr.fit(X_train, y_train)

y_pred = stregr.predict(X_test)
print(rmsle_metric(y_pred, y_test))

