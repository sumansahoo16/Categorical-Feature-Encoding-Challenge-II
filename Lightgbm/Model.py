"""
Created on Thu Feb 27 2020
@author: sumansahoo16
Public Score : 0.78566
"""
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

data = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

data.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

target = data['target']
train = data.drop('target',axis=1)

train['null'] = train.isna().sum(axis=1)
test['null'] = test.isna().sum(axis=1)

train = train.applymap(str)
test = test.applymap(str)

bin_var = ['bin_0','bin_1','bin_2','bin_3','bin_4']
nom_cols = ['nom_{}'.format(i) for i in range(10)]
ord_cols = ['ord_{}'.format(i) for i in range(6)]
target_cols = bin_var + nom_cols + ord_cols + ['day' , 'month']

encoded = pd.DataFrame([])
for tr_in,fold_in in StratifiedKFold(n_splits=12, shuffle=True).split(train, target):
    encoder = ce.TargetEncoder(cols = target_cols, smoothing=0.2)
    encoder.fit(train.iloc[tr_in,:],target.iloc[tr_in])
    encoded = encoded.append(encoder.transform(train.iloc[fold_in,:]),ignore_index=False)
encoder = ce.TargetEncoder(cols = target_cols,smoothing=0.2)
encoder.fit(train,target)
test = encoder.transform(test)
train = encoded.sort_index()

best_params = {     'learning_rate': 0.05,
                    'feature_fraction': 0.1,
                    'min_data_in_leaf' : 12,
                    'max_depth': 3,
                    'reg_alpha': 1,
                    'reg_lambda': 1,
                    'objective': 'binary',
                    'metric': 'auc',
                    'n_jobs': -1,
                    'n_estimators' : 5000,
                    'feature_fraction_seed': 42,
                    'bagging_seed': 42,
                    'boosting_type': 'gbdt',
                    'verbose': 1,
                    'is_unbalance': True,
                    'boost_from_average': False}
n_splits = 12
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=16)

roc_auc = list()
average_precision = list()
oof = np.zeros(len(train))
cv_test_preds = np.zeros(len(test))

for train_idx, test_idx in skf.split(train, target):
    X_train, y_train = train.iloc[train_idx, :], target[train_idx]
    X_test, y_test = train.iloc[test_idx, :], target[test_idx]
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set = [(X_train, y_train),( X_test, y_test)],
              verbose = 1000,
              eval_metric = 'auc',
              early_stopping_rounds = 1000)
    
    preds = model.predict_proba(X_test)
    oof[test_idx] = preds[:,1]
    
    cv_test_preds += model.predict_proba(test)[:,1] / n_splits
    
    roc_auc.append(roc_auc_score(y_true=y_test, y_score=preds[:,1]))
    average_precision.append(average_precision_score(y_true=y_test, y_score=preds[:,1]))
    
submission = pd.read_csv('sample_submission.csv', index_col='id')
submission['target'] = y_pred
submission.to_csv('submission.csv')
