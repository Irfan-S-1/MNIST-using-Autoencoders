# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 12:07:03 2021

@author: Irfan Sheikh
"""
import os
os.chdir("D:\Data Science\Kaggle\MNIST")
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

df=pd.read_csv("train.csv")
df.head()
df.info()

x_data = df.iloc[:, 1:].to_numpy()
y_data = df.iloc[:, 0].to_numpy()

nor_x_data=x_data.astype('float32')/255.

x_train,x_test,y_train,y_test=train_test_split(nor_x_data,y_data,test_size=0.2)

plt.imshow(x_train[0].reshape(28, 28))
print (y_train[0])

#Creating Autoencoder#

input_layer=Input(shape=(x_train.shape[1],))
encoded=Dense(units=64,activation="relu")(input_layer)
decoded=Dense(units=x_train.shape[1],activation="sigmoid")(encoded)
auto_encoder = Model(input_layer, decoded)

encoder = Model(input_layer, encoded)


encoded_input = Input(shape=(64, ))
decoder_layer = auto_encoder.layers[-1](encoded_input)
decoder = Model(encoded_input, decoder_layer)

auto_encoder.summary()
encoder.summary()
decoder.summary()

callbacks = EarlyStopping(monitor='val_loss',
                          min_delta=0.0001,
                          patience=5,
                          restore_best_weights=True)

auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

history = auto_encoder.fit(x_train, x_train,
                           epochs=100, batch_size=256,
                           validation_data=(x_test, x_test),
                           shuffle=True)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

encoded = encoder.predict(x_test)
decoded = decoder.predict(encoded)

fig = plt.figure(figsize=(16, 8))

n_plots = 10
n_rows = int(n_plots / 2)

for j in range(n_plots):
    
    fig.add_subplot(n_rows, 6, 3*j+1)
    plot_tmp = plt.imshow(x_test[j].reshape(28, 28))
    plt.xticks([])
    plt.yticks([])
    
    fig.add_subplot(n_rows, 6, 3*j+2)
    plot_tmp = plt.imshow(encoded[j].reshape(1, 64))
    plt.xticks([])
    plt.yticks([])
    
   
    fig.add_subplot(n_rows, 6, 3*j+3)
    plot_tmp = plt.imshow(decoded[j].reshape(28, 28))
    plt.xticks([])
    plt.yticks([])
    
plt.show()