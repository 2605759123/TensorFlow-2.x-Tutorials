import tensorflow as tf

import matplotlib.pyplot as plt


def func(x):
    """

    :param x: [b, 2]
    :return:
    """
    z = tf.math.sin(x[...,0]) + tf.math.sin(x[...,1])

    return z

#tf.linspace(start, end, num)：这个函数主要的参数就这三个，start代表起始的值，end表示结束的值，num表示在这个区间里生成数字的个数，生成的数组是等间隔生成的。start和end这两个数字必须是浮点数，不能是整数，如果是整数会出错的，请注意！
x = tf.linspace(0., 2*3.14, 500)
y = tf.linspace(0., 2*3.14, 500)
# [500, 500]
point_x, point_y = tf.meshgrid(x, y)
# [500, 500, 2]
points = tf.stack([point_x, point_y], axis=2)
# points = tf.reshape(points, [-1, 2])
print('points:', points.shape)
z = func(points)
print('z:', z.shape)

plt.figure('plot 2d func value')
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(point_x, point_y, z)
plt.colorbar()
plt.show()