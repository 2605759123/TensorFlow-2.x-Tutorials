import tensorflow as tf
import numpy as np
print(tf.__version__)
y = tf.linspace(-2., 2, 5)
x = tf.linspace(-2., 2, 5)

point_x, point_y = tf.meshgrid(x, y)
