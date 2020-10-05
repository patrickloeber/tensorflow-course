import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize
x_train, x_test = x_train / 255.0, x_test / 255.0


# Feed forward neural network
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
])

# config
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001) # "adam"
metrics = [keras.metrics.SparseCategoricalAccuracy()] # "accuracy"

# compile
model.compile(loss=loss, optimizer=optim, metrics=metrics)

# fit/training
model.fit(x_train, y_train, batch_size=64, epochs=5, shuffle=True, verbose=2)

print("Evaluate:")
model.evaluate(x_test,  y_test, verbose=2)

# 1) Save whole model
# two formats: SavedModel or HDF5
model.save("nn")  # no file ending = SavedModel
model.save("nn.h5")  # .h5 = HDF5

new_model = keras.models.load_model("nn.h5")

# 2) save only weights
model.save_weights("nn_weights.h5")

# initilaize model first:
# model = keras.Sequential([...])
model.load_weights("nn_weights.h5")

# 3) save only architecture, to_json
json_string = model.to_json()

with open("nn_model.json", "w") as f:
    f.write(json_string)

with open("nn_model.json", "r") as f:
    loaded_json_string = f.read()

new_model = keras.models.model_from_json(loaded_json_string)
print(new_model.summary())


