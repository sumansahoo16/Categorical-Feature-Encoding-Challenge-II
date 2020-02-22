"""
Created on Mon Feb 17 2020
@author: sumansahoo16
Public Score : 0.78554
"""

import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

target = data['target']
train = data.drop('target',axis=1)

bin_var = ['bin_0','bin_1','bin_2','bin_3','bin_4']
nom_cols = ['nom_{}'.format(i) for i in range(10)]
ord_cols = ['ord_{}'.format(i) for i in range(6)]
encoding_cols = bin_var + nom_cols + ord_cols + ['day' , 'month']
################### TARGET-ENCODING ############################
encoded = pd.DataFrame([])
for tr_in,fold_in in StratifiedKFold(n_splits=12, shuffle=True).split(train, target):
    encoder = ce.TargetEncoder(cols = encoding_cols, smoothing=0.2)
    encoder.fit(train.iloc[tr_in,:],target.iloc[tr_in])
    encoded = encoded.append(encoder.transform(train.iloc[fold_in,:]),ignore_index=False)
    encoder = ce.TargetEncoder(cols = encoding_cols,smoothing=0.2)
    encoder.fit(train,target)
    test_ = encoder.transform(test)
    train_ = encoded.sort_index()
#################################################################
X_train, X_test, y_train, y_test = train_test_split(train_, target, test_size=0.005, random_state=16)


model = Sequential()
model.add(Dense(64, input_dim=23, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5,  min_lr=0.00001)
checkpoint = ModelCheckpoint("best_weights.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

epochs = 25
batch_size = 128

model.fit(X_train,y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_test,y_test), callbacks=[checkpoint, learning_rate_reduction])

y_pred = model.predict_proba(test_)
submission = pd.read_csv('sample_submission.csv', index_col='id')
submission['target'] = y_pred
submission.to_csv('submission.csv')

