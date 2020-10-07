import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

#  a           a             a     b          a
#  |           |              \    /        /   \
#  b           b                c          b     c
#  |          /  \              |          \     /
#  c         c    d             d             d

# model: Sequential: one input, one output
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
])

print(model.summary())

# create model with functional API
# Advantages:
#   - Models with multiple inputs and outputs
#   - Shared layers
#   - Extract and reuse nodes in the graph of layers
#   - Model are callable like layers (put model into sequential)
# start by creating an Input node
inputs = keras.Input(shape=(28,28))

flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(128, activation='relu')
dense2 = keras.layers.Dense(10)

x = flatten(inputs)
x = dense1(x)
outputs = dense2(x)

# or with multiple outputs
#dense2_2 = keras.layers.Dense(1)
#outputs2 = dense2_2(x)
#outputs = [output, outputs2]

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

print(model.summary())

# convert functional to sequential model
# only works if the layers graph is linear.
new_model = keras.models.Sequential()
for layer in model.layers:
    new_model.add(layer)
    
# convert sequential to functional
inputs = keras.Input(shape=(28,28))
x = new_model.layers[0](inputs)
for layer in new_model.layers[1:]:
    x = layer(x) 
outputs = x

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
print(model.summary())


# access inputs, outputs for model
# access input + output for layer
# access all layers
inputs = model.inputs
outputs = model.outputs
print(inputs)
print(outputs)

input0 = model.layers[0].input
output0 = model.layers[0].output
print(input0)
print(output0)

# Example: Transfer Learning:
base_model = keras.applications.VGG16()

x = base_model.layers[-2].output
new_outputs = keras.layers.Dense(1)(x)

new_model = keras.Model(inputs=base_model.inputs, outputs=new_outputs)

