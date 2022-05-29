#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the initial libraries needed
import os
import numpy as np 
import pandas as pd 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# # Import dataset

# In[2]:


dataframe=pd.read_csv("./heart.csv")
dataframe.head()


# ###  Imports needed for pre processing of data

# In[3]:


from sklearn import preprocessing
import matplotlib 
matplotlib.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder


# ### List the type of attributes

# In[4]:


dataframe.dtypes


# ###### Convert from object to string

# In[5]:


dataframe[dataframe.select_dtypes(include="object").columns]=dataframe[dataframe.select_dtypes(include="object").columns].astype("string")


# In[6]:


categorical_columns=dataframe.select_dtypes("string").columns.to_list()
numerical_col=dataframe.select_dtypes("int64").columns.to_list()
numerical_col=numerical_col+dataframe.select_dtypes("float64").columns.to_list()
dataframe.describe().T


# In[7]:


px.imshow(dataframe.corr(),title="Correlation of Heart Disease")


# MaxHr has a negative correlation with heart diseas 
# <br />
# Cholesterol has a negative correlation with heart diseas
# <br />
# Oldpeak has a positive correlation with heart diseas
# <br />
# FastingBS has a positive correlation with heart diseas
# <br />
# RestingBP has a positive correlation with heart diseas
# 

# # Data Preprocessing

# In[8]:


#Check for null values
dataframe.info()
categorical_columns=dataframe.select_dtypes("string").columns.to_list()
numerical_col=dataframe.select_dtypes("int64").columns.to_list()
numerical_col=numerical_col+dataframe.select_dtypes("float64").columns.to_list()


# In[9]:


le = LabelEncoder()
# select numerical features
numerical_features = dataframe.select_dtypes(include=['int64', 'float64'])
# apply label encoding
numerical_features = numerical_features.apply(LabelEncoder().fit_transform)
numerical_features.head()


# In[ ]:





# ##### Feature Scaling

# In[10]:


x = pd.DataFrame({
    # Distribution with lower outliers
    'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(1, 2, 25)]),
    # Distribution with higher outliers
    'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(50, 2, 25)]),
})
np.random.normal

scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(x)
robust_df = pd.DataFrame(robust_df, columns =['x1', 'x2'])


scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(x)
standard_df = pd.DataFrame(standard_df, columns =['x1', 'x2'])
 
scaler = preprocessing.MinMaxScaler()
minmax_df = scaler.fit_transform(x)
minmax_df = pd.DataFrame(minmax_df, columns =['x1', 'x2'])
 
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize =(20, 5))
ax1.set_title('Before Scaling')
 
sns.kdeplot(x['x1'], ax = ax1, color ='r')
sns.kdeplot(x['x2'], ax = ax1, color ='b')
ax2.set_title('After Robust Scaling')
 
sns.kdeplot(robust_df['x1'], ax = ax2, color ='red')
sns.kdeplot(robust_df['x2'], ax = ax2, color ='blue')
ax3.set_title('After Standard Scaling')
 
sns.kdeplot(standard_df['x1'], ax = ax3, color ='black')
sns.kdeplot(standard_df['x2'], ax = ax3, color ='g')
ax4.set_title('After Min-Max Scaling')
 
sns.kdeplot(minmax_df['x1'], ax = ax4, color ='black')
sns.kdeplot(minmax_df['x2'], ax = ax4, color ='g')
plt.show()
 


# In[11]:


# As we will be using both types of approches for demonstration lets do First Label Ecoding 
# which will be used with Tree Based Algorthms
df_tree = dataframe.apply(LabelEncoder().fit_transform)
df_tree.head()


# In[12]:


## Creaeting one hot encoded features for working with non tree based algorithms 
df_nontree=pd.get_dummies(dataframe,columns=categorical_columns,drop_first=False)
target="HeartDisease"
y=df_nontree[target].values
df_nontree.drop("HeartDisease",axis=1,inplace=True)
df_nontree=pd.concat([df_nontree,dataframe[target]],axis=1)
df_nontree.head()


# In[13]:


# separet features and target
features = df_nontree.drop(['HeartDisease'], axis=1)
labels = df_nontree['HeartDisease']
features.head()


# In[14]:


labels.head()


# In[15]:


# train test split
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)


# # XGBOOST

# In[16]:


# model building xgboost
from xgboost import XGBClassifier
xgboost = XGBClassifier()
xgboost.fit(train_features, train_labels)
# predict
xgb_pred = xgboost.predict(test_features)
# accuracy
from sklearn.metrics import accuracy_score
xgb_accuracy = accuracy_score(test_labels, xgb_pred)
print('XGBoost Accuracy:', xgb_accuracy)


# In[17]:


from sklearn.metrics import classification_report
print(classification_report(test_labels, xgb_pred))


# In[18]:


# Feature importance for xgboost
feat_importances = pd.Series(xgboost.feature_importances_, index=features.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()


# # Random Forest

# In[19]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);
#predict
rf_pred = rf.predict(test_features)
# accuracy
rf_accuracy = round(rf.score(test_features,test_labels),3)
print('Random Forest Accuracy: ', rf_accuracy)


# In[20]:


# Feature importance for Random Forest
feat_importances = pd.Series(rf.feature_importances_, index=features.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()


# # Naive Bayes

# In[21]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
gnb = GaussianNB()
gnb.fit(train_features, train_labels)
gnb_pred = gnb.predict(test_features)
gnb_accuracy = metrics.accuracy_score(test_labels, gnb_pred)
print('Naive Bayes Accuracy: {0:0.3f}'. format(gnb_accuracy))


# In[22]:


print(classification_report(test_labels, gnb_pred))


# # LIGHT GBM

# In[23]:


#Importing Required Packages
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
lgb_df_tree = df_tree
lgb_df_tree = lgb_df_tree.rename(columns={'HeartDisease':'Label'})
lgb_df_tree['Label'].value_counts()

lgbm_y = lgb_df_tree["Label"]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
lgbm_y_axis = labelencoder.fit_transform(lgbm_y) 

lgbm_x_axis= lgb_df_tree.drop(["Label"], axis=1)

#Defining Features
lgbm_feature_names = np.array(lgbm_x_axis.columns)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(lgbm_x_axis)
lgbm_x_axis = scaler.transform(lgbm_x_axis)



lgbm_params = {'learning_rate':0.005, 'boosting_type':'gbdt',   
              'objective':'poisson',
              'metric':['poisson'],
              'num_leaves':10000,
              'max_depth':100}


##Spliting dataframe into train data and 10 % test data
from sklearn.model_selection import train_test_split
lgbm_X_train, lgbm_X_test, lgbm_y_train, lgbm_y_test = train_test_split(lgbm_x_axis, lgbm_y_axis, test_size=0.1, random_state=42)

d_train = lgb.Dataset(lgbm_X_train, label=lgbm_y_train)

clf = lgb.train(lgbm_params, d_train, 50)

y_prediction=clf.predict(lgbm_X_test)

for i in range(0, lgbm_X_test.shape[0]):
    if y_prediction[i]>=.5:       # setting threshold to .5
       y_prediction[i]=1
    else:  
       y_prediction[i]=0
    
cm_lgbm = confusion_matrix(lgbm_y_test, y_prediction)
sns.heatmap(cm_lgbm, annot=True)
lgbm_accuracy = metrics.accuracy_score(lgbm_y_test, y_prediction)

print ("Accuracy with LGBM = ",lgbm_accuracy)


# In[24]:


print(classification_report(lgbm_y_test, y_prediction))


# # Comparing Models

# In[25]:


plotdata = pd.DataFrame({

    "XGBoost":[xgb_accuracy],

    "Random Forest":[rf_accuracy],

    "Naive Bayes":[gnb_accuracy],
    
    "LIGHT GBM" : [lgbm_accuracy]},

    index=["Accuracy"])

plotdata.plot(kind="bar",figsize=(15, 8))

plt.title("Model Performance")

plt.xlabel("Metrics")

plt.ylabel("Scores")


# # Tune Hyper parameters

# In[26]:


from sklearn.metrics import accuracy_score
import xgboost as xgb


# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# In[27]:


space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform('n_estimators',400, 450, 500),
        'seed': 0
    }


# In[28]:


def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( train_features, train_labels), ( test_features, test_labels)]
    clf.set_params(eval_metric="auc",
            early_stopping_rounds=10)
    clf.fit(train_features, train_labels,
            eval_set=evaluation ,verbose=False)
    

    pred = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }


# In[29]:


trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)


# In[30]:


print("The best hyperparameters are : ","\n")
print(best_hyperparams)


# # Grid Search

# In[31]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [ 500 ,700,800, 1000],
           'colsample_bytree': [0.3, 0.7]}
xgbr = xgb.XGBRegressor(seed = 20)
clf = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring='neg_mean_squared_error', 
                   verbose=1)
clf.fit(train_features,train_labels)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))


# In[32]:



xgboost_new = XGBClassifier(colsample_bytree= 0.3, learning_rate= 0.01, max_depth = 3, n_estimators = 700)
xgboost_new.fit(train_features, train_labels)
best_y_pred=xgboost_new.predict(test_features)

print(classification_report(test_labels, xgb_pred))
print(classification_report(test_labels, best_y_pred))


# In[33]:


# Feature importance for xgboost
feat_importances = pd.Series(xgboost_new.feature_importances_, index=features.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()

