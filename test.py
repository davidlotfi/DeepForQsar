import keras
import tensorflow as tf
import pandas as pd
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense,Activation


#load data
data=pd.read_csv('ic.csv', sep='\t',  engine='python',
    na_values=['NA','?'])

print(len(data))
#concatination colone...



#premier definie architecture de modele et le nomber de couche

model = Sequential()
model.add(Dense(32,input_dim=784))
model.add(Activation('relu'))
print('Define modele avec 2 couche ')


#compile le modele

model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
              metrics=['accuracy'])
print('compile modele')


#fit model in training data

#model.fit(data,epochs=10, batch_size=32)