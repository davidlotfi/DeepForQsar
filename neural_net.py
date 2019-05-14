from keras.models import Sequential
from Keras.layers import Dense ,Dropout

def generate_model(input_shape):
    model=Sequential()

    model.add(Dense(200,activation='relu',input_shape=input_shape))
    model.add(Dropout())
    model.add(Dense(200,activation='relu',input_shape=input_shape))
    model.add(Dropout())


    model.compile(optimizer='rmsprop',loss='mean_squared_error',
                  matric=['accuracy'])


    return model

