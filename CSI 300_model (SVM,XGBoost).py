#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('dataset4.csv')



#Split data set. Since stock data is time series data, all data cannot be scrambled.
#Meanwhile, the test set accounts for 20%
test_ratio=0.2
test_size=int(len(df) * test_ratio)
test_set=df[-test_size:]
print(test_set)




# The rest is for training
train_set=df[:-test_size]
print(train_set)





#n fold cross validation split training set and verification set
#time series split according to time series characteristics
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
'''df1=df.values
X=df1[:,1:]
Y=df1[:,0]
'''
train_set1=train_set.values
X=train_set1[:,1:]
Y=train_set1[:,0]

splits = TimeSeriesSplit(n_splits=5)


for train_index, test_index in splits.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print('Observations: %d' % (len(X_train) + len(X_test)))
    print('Training Observations: %d' % (len(X_train)))
    print('Testing Observations: %d' % (len(X_test)))



test_set1=test_set.values
x_test=test_set1[:,1:]
y_test=test_set1[:,0]





import numpy as np
import sklearn.svm as svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
parameters={'kernel':['rbf'],'C':[0.1,1,10],'gamma':[0.1,1,10]}
svc = svm.SVC()
svm_model = GridSearchCV(svc,parameters,cv=splits,scoring='accuracy')
svm_model.fit(X,Y)
print(svm_model.best_params_)
svm_model.score(x_test,y_test)





import numpy as np
import sklearn.svm as svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
parameters={'kernel':['poly'],'C':[0.1,1,10],'degree':[1,2,3]}
svc = svm.SVC()
svm_model = GridSearchCV(svc,parameters,cv=splits,scoring='accuracy')
svm_model.fit(X,Y)
print(svm_model.best_params_)
svm_model.score(x_test,y_test)





#The training set, test set and verification set are divided according to the time series
# results of SVM
test_ratio=0.2
test_size=int(len(df) * test_ratio)
test_set=df[-test_size:]
print(test_set)

valid_ratio=0.2
valid_size=int(len(df) * test_ratio)
valid_set=df[-test_size-valid_size:-test_size]
print(valid_set)

train_set=df[:-test_size-valid_size]
print(train_set)

train_set1=train_set.values
x_train=train_set1[:,1:24]
y_train=train_set1[:,0]

test_set1=test_set.values
x_test=test_set1[:,1:24]
y_test=test_set1[:,0]

valid_set1=valid_set.values
x_valid=valid_set1[:,1:24]
y_valid=valid_set1[:,0]




'''
standardScaler=StandardScaler()
standardScaler.fit(x_train)

standardScaler.mean_

standardScaler.scale_

standardScaler.transform(x_train) #normalization x_train

x_train=standardScaler.transform(x_train)

x_test=standardScaler.transform(x_test) #normalization x_test

x_valid=standardScaler.transform(x_valid)

'''

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)

test_set['target'].value_counts()

valid_set['target'].value_counts()

train_set['target'].value_counts()

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)





from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



#支持向量机
from sklearn.svm import SVC
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
#svm_model=SVC(kernel='poly',C=1,degree=2)
#svc = svm.SVC(kernel='rbf',C=10,gamma=0.1)






svm_model.fit(x_train,y_train.astype('int'))





results2 = svm_model.predict(x_train)
print("svm_train_accuracy:", 1 - sum((y_train - results2)**2)/len(results2))
results2 = svm_model.predict(x_valid)
print("svm_valid_accuracy:", 1 - sum((y_valid - results2)**2)/len(results2))
results2 = svm_model.predict(x_test)
print("svm_test_accuracy:", 1 - sum((y_test - results2)**2)/len(results2))





#The results of XGBoost,set random_state differently and the results will be slightly different
import skopt
from skopt import gp_minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
params_fixed = {
        'objective': 'binary:logistic',
        'silent': 1
    }
space = {
       'max_depth': (3,4,5,6,7,8),
        'learning_rate': (10**-4,10**-3,10**-2),
        'n_estimators': (10,200),
        'min_child_weight': (1,10),
        'subsample': (0.1, 0.9),
        'colsample_bytree': (0.1,0.9),
        'gamma':(0.1,0.2,0.3,0.4,0.5,0.6),
     #'reg_lambda':(0.05, 0,0.1, 1, 2, 3),
       # 'reg_alpha':(0.05, 0,0.1, 1, 2, 3)
    }
reg = XGBClassifier(**params_fixed)

def objective(params):
        """ Wrap a cross validated inverted `accuracy` as objective func """
        reg.set_params(**{k: p for k, p in zip(space.keys(), params)})
        return 1-np.mean(cross_val_score(
            reg, X, Y, cv=splits, n_jobs=-1,
            scoring='accuracy')
        )
    
res_gp = gp_minimize(objective, space.values(), n_calls=50,random_state=277)
best_hyper_params = {k: v for k, v in zip(space.keys(), res_gp.x)}

params = best_hyper_params.copy()
params.update(params_fixed)

clf = XGBClassifier(**params)
clf.fit(X,Y)
y_test_preds = clf.predict(x_test)

plot_importance(clf,max_num_features=10)


display(pd.crosstab(
        pd.Series(y_test, name='Actual'),
        pd.Series(y_test_preds, name='Predicted'),
        margins=True
    ))
print ('Accuracy: {0:.5f}'.format(accuracy_score(y_test, y_test_preds)))





params = best_hyper_params.copy()
params.update(params_fixed)
print(params)



# xgboost
from xgboost import XGBClassifier
xgbc_model=XGBClassifier(max_depth=8,
                      learning_rate=0.01,
                      n_estimators=10,
                      silent=1,
                      objective='binary:logistic',
                      gamma=0.6,
                      min_child_weight=1,
                      subsample=0.9,
                      colsample_bytree=0.9,
                      seed=277
                        )





xgbc_model.fit(x_train,y_train.astype('int'))





results3= xgbc_model.predict(x_train)
print("xgbc_train_accuracy:", 1 - sum((y_train - results3)**2)/len(results3))
results3 =xgbc_model.predict(x_valid)
print("xgbc_valid_accuracy:", 1 - sum((y_valid - results3)**2)/len(results3))
results3 =xgbc_model.predict(x_test)
print("xgbc_test_accuracy:", 1 - sum((y_test - results3)**2)/len(results3))







