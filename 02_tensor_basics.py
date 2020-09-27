import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

"""
Everything in TensorFlow is based on Tensor operations.
Tensors are (kind of) like np.arrays.
All tensors are immutable: you can never update the contents of a
tensor, only create a new one.

 - nd-arrays (1d, 2d, or even 3d and higher)
 - GPU support
 - Computational graph / Track gradients / Backpropagation
 - Immutable!
"""
# 1. create tensors
# scalar, rank-0 tensor
x = tf.constant(4)
print(x)

x = tf.constant(4, shape=(1,1), dtype=tf.float32)
print(x)

# vector, rank-1
x = tf.constant([1,2,3])
print(x)

# matrix, rank-2
x = tf.constant([[1,2,3], [4,5,6]])
print(x)

x = tf.ones((3,3))
print(x)

x = tf.zeros((3,3))
print(x)

x = tf.eye(3)
print(x)

x = tf.random.normal((3,3), mean=0, stddev=1)
print(x)

x = tf.random.uniform((3,3), minval=0, maxval=1)
print(x)

x = tf.range(10)
print(x)

# 2. cast:
x = tf.cast(x, dtype=tf.float32)
print(x)

# 3. operations, elementwise
x = tf.constant([1,2,3])
y = tf.constant([4,5,6])

z = tf.add(x,y)
z = x + y
print(z)

z = tf.subtract(x,y)
z = x - y
print(z)

z = tf.divide(x,y)
z = x / y
print(z)

z = tf.multiply(x,y)
z = x * y
print(z)

# dot product
z = tf.tensordot(x,y, axes=1)
print(z)

# elementwise exponentiate
z = x ** 3
print(z)

# matrix multiplication (shapes must match: number of columns A = number of rows B)
x = tf.random.normal((2,2)) # 2,3
y = tf.random.normal((3,4)) # 3,4

z = tf.matmul(x,y)
z = x @ y
print(z)

# 4. indexing, slicing
x = tf.constant([[1,2,3,4],[5,6,7,8]])
print(x[0])
print(x[:, 0]) # all rows, column 0
print(x[1, :]) # row 1, all columns
print(x[1,1]) # element at 1, 1

# 5. reshape
x = tf.random.normal((2,3))
print(x)
x = tf.reshape(x, (3,2))
print(x)

x = tf.reshape(x, (-1,2))
print(x)

x = tf.reshape(x, (6))
print(x)

# 6. numpy
x = x.numpy()
print(type(x))

x = tf.convert_to_tensor(x)
print(type(x))
# -> eager tensor = evaluates operations immediately
# without building graphs

# string tensor
x = tf.constant("Patrick")
print(x)

x = tf.constant(["Patrick", "Max", "Mary"])
print(x)

# Variable
# A tf.Variable represents a tensor whose value can be
# changed by running ops on it
# Used to represent shared, persistent state your program manipulates
# Higher level libraries like tf.keras use tf.Variable to store model parameters.
b = tf.Variable([[1.0, 2.0, 3.0]])
print(b)
print(type(b))
