"""
Created on Sat Feb 12 2020
@author: sumansahoo16
Public Score : 0.78584
"""

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

target = data['target']
train = data.drop('target',axis=1)

train = train.applymap(str)
test = test.applymap(str)

encoding_cols = train.columns
encoded = pd.DataFrame([])
for tr_in,fold_in in StratifiedKFold(n_splits=12, shuffle=True).split(train, target):
    encoder = ce.TargetEncoder(cols = encoding_cols, smoothing=0.2)
    encoder.fit(train.iloc[tr_in,:],target.iloc[tr_in])
    encoded = encoded.append(encoder.transform(train.iloc[fold_in,:]),ignore_index=False)
encoder = ce.TargetEncoder(cols = encoding_cols,smoothing=0.2)
encoder.fit(train,target)
test = encoder.transform(test)
train = encoded.sort_index()

best_params = {'bagging_temperature': 0.8,
               'depth': 5,
               'iterations': 1000,
               'l2_leaf_reg': 30,
               'learning_rate': 0.05,
               'random_strength': 0.8}

n_splits = 12
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=16)

roc_auc = list()
average_precision = list()
oof = np.zeros(len(train))
cv_test_preds = np.zeros(len(test))
best_iteration = list()

for train_idx, test_idx in skf.split(train, target):
    X_train, y_train = train.iloc[train_idx, :], target[train_idx]
    X_test, y_test = train.iloc[test_idx, :], target[test_idx]
    
    Train = Pool(data=X_train, 
             label=y_train,            
             feature_names=list(X_train.columns))

    val = Pool(data=X_test, 
               label=y_test,
               feature_names=list(X_test.columns))

    catb = CatBoostClassifier(**best_params,
                          loss_function='Logloss',
                          eval_metric = 'AUC',
                          nan_mode='Min',
                          thread_count=4,
                          verbose = False)
    
    catb.fit(Train,
             verbose_eval=100, 
             early_stopping_rounds=50,
             eval_set=val,
             use_best_model=True,
             plot=False)
    
    best_iteration.append(catb.best_iteration_)
    preds = catb.predict_proba(X_test)
    oof[test_idx] = preds[:,1]
    
    Xt_pool = Pool(data=test[list(X_train.columns)],
               feature_names=list(X_train.columns))
    
    cv_test_preds += catb.predict_proba(Xt_pool)[:,1] / n_splits
    
    roc_auc.append(roc_auc_score(y_true=y_test, y_score=preds[:,1]))
    average_precision.append(average_precision_score(y_true=y_test, y_score=preds[:,1]))

y_pred = model.predict_proba(test_)
submission = pd.read_csv('sample_submission.csv', index_col='id')
submission['target'] = y_pred
submission.to_csv('submission.csv')
