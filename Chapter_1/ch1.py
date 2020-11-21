# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:42:49 2020

@author: Seokwoojoon
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([
    Dense(1, input_shape=[1])]
    )
model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.arange(-1,5,1, dtype = np.float32)

def fun(x):
    x = x*2-1
    return x
y = np.array(list(map(fun,x)), dtype = np.float32)
model.fit(x,y,epochs = 500)
print(model.predict([10.0]))

#%%
# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1,2,3,4,5,6,7], dtype=np.float32)
    ys =np.array([100,150,200,250,300,350,400], dtype=np.float32)  
    model = keras.Sequential(keras.layers.Dense(1,input_shape=[1]))  
    model.compile(optimizer='sgd', loss = 'mean_squared_error')
    model.fit(xs,ys,epochs = 500)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)