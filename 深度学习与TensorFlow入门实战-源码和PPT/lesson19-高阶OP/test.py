import tensorflow as tf

a = tf.random.normal([3, 3])

tf.print(a)


mask = a > 0
tf.print(mask)


indices=tf.where(mask)

tf.print(indices)

tf.print(tf.gather_nd(a,indices))


