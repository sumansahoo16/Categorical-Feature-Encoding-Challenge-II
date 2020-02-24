"""
Created on Wed Feb 19 2020
@author: sumansahoo16
Public Score : 0.78568
"""

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

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
test_ = encoder.transform(test)
train_ = encoded.sort_index()

best_params = {'bagging_temperature': 0.8,
               'depth': 5,
               'iterations': 1000,
               'l2_leaf_reg': 30,
               'learning_rate': 0.05,
               'random_strength': 0.8}

model = CatBoostClassifier(**best_params,
                          loss_function='Logloss',
                          eval_metric = 'AUC',
                          nan_mode='Min',
                          thread_count=4,
                          verbose = False)

model.fit(train_,target,verbose_eval=100,early_stopping_rounds=50,use_best_model=True,plot=False)

y_pred = model.predict_proba(test_)
submission = pd.read_csv('sample_submission.csv', index_col='id')
submission['target'] = y_pred
submission.to_csv('submission.csv')

