import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# 28, 28 -> treat image as sequence
# input_size=28
# seq_length=28


# RNN Outputs and states
# ----------------------
# By default, the output of a RNN layer contains a single vector per sample.
# This vector is the RNN cell output corresponding to the last timestep, 
# containing information about the entire input sequence. 
# The shape of this output is (N, units)
# 
# A RNN layer can also return the entire sequence of outputs 
# for each sample (one vector per timestep per sample), 
# if you set return_sequences=True. The shape of this output 
# is (N, timesteps, units).


# model
model = keras.models.Sequential()
model.add(keras.Input(shape=(28,28))) # seq_length, input_size
#model.add(layers.SimpleRNN(128, return_sequences=True, activation='relu')) # N, 28, 128
model.add(layers.LSTM(128, return_sequences=False, activation='relu')) # N, 128
model.add(layers.Dense(10))
print(model.summary())

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

# evaulate
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

    

