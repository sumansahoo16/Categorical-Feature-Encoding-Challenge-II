"""
Created on Mon Feb 23 2020
@author: sumansahoo16
Public Score : 0.76418
"""

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

target = data['target']
train = data.drop('target',axis=1)

train['missing_count'] = train.isnull().sum(axis=1)
test['missing_count'] = test.isnull().sum(axis=1)

bin_var = ['bin_0','bin_1','bin_2','bin_3','bin_4']
nom_cols = ['nom_{}'.format(i) for i in range(10)]
ord_cols = ['ord_{}'.format(i) for i in range(6)]
target_cols = bin_var + nom_cols + ord_cols + ['day' , 'month'] + ['ord_5_1', 'ord_5_2']

train['ord_5_1'] = train['ord_5'].apply(lambda x: x[0] if type(x) == str else np.nan)
train['ord_5_2'] = train['ord_5'].apply(lambda x: x[1] if type(x) == str else np.nan)
test['ord_5_1'] = test['ord_5'].apply(lambda x: x[0] if type(x) == str else np.nan)
test['ord_5_2'] = test['ord_5'].apply(lambda x: x[1] if type(x) == str else np.nan)

popl_Dict = {'Russia':142893540,
             'Canada':33098932,
             'Finland':5231372,
             'Costa Rica':4075261,
             'China':1313973713,
             'India':1095351995}

area_Dict = {'Russia':17075200,
             'Canada':9984670,
             'Finland':338145,
             'Costa Rica':51100,
             'China':9596960,
             'India':3287590}

train['country_popl'] = train['nom_3'].map(popl_Dict)
train['country_area'] = train['nom_3'].map(area_Dict)
test['country_popl'] = test['nom_3'].map(popl_Dict)
test['country_area'] = test['nom_3'].map(area_Dict)

train['country_popl'].fillna(train['country_popl'].median(),inplace = True)
train['country_area'].fillna(train['country_area'].median(),inplace = True)
test['country_popl'].fillna(test['country_popl'].median(),inplace = True)
test['country_area'].fillna(test['country_area'].median(),inplace = True)

scaler = MinMaxScaler()

train[['country_popl','country_area']] = scaler.fit_transform(train[['country_popl','country_area']],target)
test[['country_popl','country_area']] = scaler.transform(test[['country_popl','country_area']])

train[target_cols] = train[target_cols].applymap(str)
test[target_cols] = test[target_cols].applymap(str)


 def frequency_encoding(column, df):
    frequencies = df[column].value_counts().reset_index()
    df_values = df[[column]].merge(frequencies, how='left', left_on=column, right_on='index').iloc[:,-1].values
    return df_values

freq_encoded = list()

for column in target_cols:
    train_values = frequency_encoding(column, train)
    train[column+'_counts'] = train_values
    test_values = frequency_encoding(column, test)
    test[column+'_counts'] = test_values
    freq_encoded.append(column+'_counts')

    
train[freq_encoded] = scaler.fit_transform(train[freq_encoded],target)
test[freq_encoded] = scaler.transform(test[freq_encoded])


encoded = pd.DataFrame([])
for tr_in,fold_in in StratifiedKFold(n_splits=12, shuffle=True).split(train, target):
    encoder = ce.TargetEncoder(cols = target_cols, smoothing=0.2)
    encoder.fit(train.iloc[tr_in,:],target.iloc[tr_in])
    encoded = encoded.append(encoder.transform(train.iloc[fold_in,:]),ignore_index=False)
encoder = ce.TargetEncoder(cols = target_cols,smoothing=0.2)
encoder.fit(train,target)
test_ = encoder.transform(test)
train_ = encoded.sort_index()

train[target_cols] = train_[target_cols]
test[target_cols] = test_[target_cols]

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.005, random_state=16)

X_test, X_eval, y_test, y_eval = train_test_split(X_test, y_test, test_size=0.166666666666666667,random_state=16,stratify=y_test)



input_dim = train.shape[1]
model = Sequential()
model.add(Dense(128, input_dim=input_dim, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(84, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))



optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

METRICS = [keras.metrics.BinaryAccuracy(name='acc'), keras.metrics.AUC(name='ROC')]
model.compile(optimizer = optimizer , loss = "binary_crossentropy",metrics = METRICS)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_ROC', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

checkpoint = ModelCheckpoint("best_weights.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

epochs = 25
batch_size = 128

history = model.fit(X_train,y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_test,y_test), callbacks=[checkpoint, learning_rate_reduction])

model.load_weights('best_weights.hdf5')

y_pred = model.predict_proba(test)
submission = pd.read_csv('sample_submission.csv', index_col='id')
submission['target'] = y_pred
submission.to_csv('submission.csv')
