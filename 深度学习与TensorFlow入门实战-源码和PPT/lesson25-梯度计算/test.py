import tensorflow as tf

x=tf.random.normal([2,4])
w=tf.random.normal([4,3])

b=tf.zeros([3])
y=tf.constant([2,0])

with tf.GradientTape() as tape:
    tape.watch([w,b])
    prob=tf.nn.softmax(x@w+b,axis=1)
    loss=tf.reduce_mean(tf.losses.MSE(tf.one_hot(y,depth=3),prob))

grads=tape.gradient(loss,[w,b])
tf.print(grads[0])
tf.print(grads[1])
