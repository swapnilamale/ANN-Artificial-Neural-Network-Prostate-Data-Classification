# -*- coding: utf-8 -*-
# Created on Mon Oct  4 08:23:40 2021

# neural networks - classification
# dataset: prostate.csv

# download the file from the google drive  pg21/ml/dataset/

######################################################################
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report

from keras.models import Sequential
from keras.layers import Dense
######################################################################

######################################################################
# read the file
path="C:\\Users\\Sahil\\Desktop\\Imarticus Learning -4\\DAY 64 DL Prostate data Classification\\prostate.csv"
data=pd.read_csv(path)
data
data.columns
#######################################################################

#######################################################################
# remove 'id' from features
data.drop(columns='id',inplace=True)
data.columns
#######################################################################

#######################################################################
# change the y-variable into number form
y = "diagnosis_result"
le = preprocessing.LabelEncoder()
data['target'] = le.fit_transform(data[y])

# check the mapping
data[[y,'target']].head(20)

# drop the original y-variable
data.drop(columns=y,inplace=True)
data.columns
#######################################################################

#######################################################################
# transform the data
data_std = data.copy()

ss = preprocessing.StandardScaler()
data_std.iloc[:,:] = ss.fit_transform(data_std.iloc[:,:])
data_std.head()
# replace Y-variable with the actual data
data_std['target'] = data['target']

# compare the actual and std data
data_std.head()
data.head()
#######################################################################

#######################################################################
# check the distribution of the Y-variable
data_std.target.value_counts()

'''
B -> 0 ->
M -> 1 ->
'''
#######################################################################

#######################################################################
# include the EDA code
#######################################################################

#######################################################################
# split the data
trainx,testx,trainy,testy = train_test_split(data_std.drop('target',1),
                                             data_std.target,
                                             test_size=0.2)

trainx.shape,trainy.shape
testx.shape,testy.shape
#######################################################################

#______________________________________________________________________
# build the Neural Network
classifier = Sequential()

# define the nodes
inputnode = len(trainx.columns)  # features
units = 16 # nodes in the hidden layer
outputnode = 1 # output node

# build the layers
# hidden layer 1
classifier.add(Dense(input_dim=inputnode,units=units,activation='relu',kernel_initializer='uniform'))

# hidden layer 2
classifier.add(Dense(units=units,activation='relu',kernel_initializer='uniform'))

# output layer
classifier.add(Dense(units=outputnode,activation='sigmoid',kernel_initializer='uniform'))

# compile the model
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# help(Dense)

# train the model on train set
EPOCHS = 1000

classifier.fit(trainx,trainy,batch_size=5,epochs=EPOCHS,validation_split=0.15)

# predict on test data
predy = classifier.predict(testx)
print(predy)

# predy.reshape(-1)
#____________________________________________________________________

#####################################################################
# store the results in a dataframe
df1 = pd.DataFrame({'actual':testy,'pred_prob':predy.reshape(-1), 'pred_class':0})
print(df1)

df1.pred_class[df1.pred_prob > 0.5] = 1
print(df1)

# accuracy score
print(accuracy_score(df1.actual, df1.pred_class))

# confusion matrix
pd.crosstab(df1.actual, df1.pred_class, margins=True)

# classification report
print(classification_report(df1.actual, df1.pred_class))

testy.value_counts()
######################################################################


