import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy

from tensorflow import keras

import scipy.io


import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

import matplotlib.pyplot as plt


inputs = keras.Input(shape=(400,1,), name="input")
x1f = Flatten()(inputs)
outputs = Dense(1, activation="sigmoid", name="predictions")(x1f)
model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError()],
)

model2 = tf.keras.models.clone_model(model)

model2.compile(
    optimizer=keras.optimizers.RMSprop(), 
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError()],
)


fake_labs = np.arange(0,10000)/10000


X = np.load('fakeDataCVGRegressor.npy')

X= X*2 - 1


x_train, x_test, y_train, y_test = train_test_split(
    X, fake_labs, test_size=0.2, random_state=1)

x_val = x_train[-100:]
y_val = y_train[-100:]

history = model.fit(
    x_train,
    y_train,
    batch_size=512,
    epochs=400,
    validation_data=(x_val, y_val),
)

Real_x = np.load("./cleanData.npy")
Real_x = Real_x.reshape((Real_x.shape[0],Real_x.shape[1],1))


X_labs = np.load("./cleanLabs.npy")
labelset = X_labs.reshape((X_labs.shape[0],X_labs.shape[1],1))

real_labs = labelset[:,3,0]


x_train2, x_test2, y_train2, y_test2 = train_test_split(
    Real_x, real_labs, test_size=0.2, random_state=1)


history2 = model2.fit(x_train2,y_train2,batch_size=512,epochs=400)

model.evaluate(Real_x,real_labs,batch_size=512)

model2.evaluate(X,fake_labs,batch_size=512)

fake_lab_guess_fr = model2.predict(X)

fake_lab_guess_ff = model.predict(X)

idx = np.random.randint(0, Real_x.shape[0], 10000)

fake_lab_guess_rf = model.predict(Real_x[idx])

fake_lab_guess_rr = model2.predict(Real_x[idx])

ff=np.abs(fake_lab_guess_ff-fake_labs.reshape((fake_lab_guess_ff.shape[0],1)))
fr=np.abs(fake_lab_guess_fr-fake_labs.reshape((fake_lab_guess_fr.shape[0],1)))
rr=np.abs(fake_lab_guess_rr-real_labs[idx].reshape((10000,1)))
rf=np.abs(fake_lab_guess_rf-real_labs[idx].reshape((10000,1)))

scipy.io.savemat('./toMatlab/regErrors.mat', {'ff':ff,
                                                    'fr':fr,
                                                    'rr':rr,
                                                    'rf':rf})


