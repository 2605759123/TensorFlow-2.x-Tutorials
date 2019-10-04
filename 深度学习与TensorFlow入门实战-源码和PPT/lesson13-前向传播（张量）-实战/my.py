import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow import contrib as contrib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# x: [60K,28,28]
# y: [60K]
(x,y),_=datasets.mnist.load_data()
#x [0-255] -> [0-1.0]
x=tf.convert_to_tensor(x,dtype=tf.float32) / 255.
y=tf.convert_to_tensor(y,dtype=tf.int32)

print(x.shape,y.shape,x.dtype,y.dtype)
#(60000, 28, 28) (60000,) <dtype: 'float32'> <dtype: 'int32'>
print(tf.reduce_min(x),tf.reduce_max(x))
print(tf.reduce_min(y),tf.reduce_max(y))
#创建数据集，从而一次取多个对象
train_db=tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
#创建一个迭代器：iterator
train_iter=iter(train_db)
sample = next(train_iter)
# batch: (128, 28, 28) (128,)
print('batch:',sample[0].shape,sample[1].shape)


#[b,784]->[b,512]->[b,256]->[b,10]
#[dim_in,dim_out],[dim_out]
#这里要封装成tf.Variable，因为tf.GradientTape这里只会记录tf.Variable类型的数据
#这里如果不指定stddev（方差），会出现“梯度爆炸”的情况（方差改为0.1）
# w1 = tf.Variable(tf.random.truncated_normal([784,256]))
#tf.random.truncated_normal：默认：均值为0，方差为1
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))

w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

#学习速率 learnrate 0.001
lr=1e-3;

for epoch in range(10):#iterate db for 10 对数据集迭代10次
    for step,(x,y) in enumerate(train_db):#for every batch 指对一次循环，对这整个数据集的迭代进度
        # h1 = x@w1+b1
        #y: [128]
        #[b,28,28] => [b,28*28]
        x = tf.reshape(x,[-1,28*28])

        #tf.Variable
        with tf.GradientTape() as tape:
            #x:[b,28*28]
            #h1 = x@w1 + b1
            # [b,784]@[784,256] + [256] => [b,256] + [256] => [b,256] + [b,256]
            h1 = x@w1 + tf.broadcast_to(b1,[x.shape[0],256])
            h1 = tf.nn.relu(h1)
            #[b,256] => [b,128]
            #不加broadcast也会自动转换
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # [b,128] => [b,10]
            out=h2@w3 + b3

            #compute loss
            #out: [b,10]
            #y: [b] => [b,10]
            y_onehot = tf.one_hot(y,depth=10)

            #mse = mean(sum(y-out)^2) 均方差
            #[b,10]
            loss = tf.square(y_onehot - out)
            #mean:scalar
            loss=tf.reduce_mean(loss)

        #compute gradients
        grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])

        #w1 = w1- lr * w1 grad

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        #这里，w1(tf.Variable),但是相减后，会返回tf.tensor类型
        #所以要原地更新，保持数据类型不变
        #w1=w1 - lr * grads[0]
        # b1=b1 - lr * grads[1]
        # w2=w2 - lr * grads[2]
        # b2=b2 - lr * grads[3]
        # w3=w3 - lr * grads[4]
        # b3=b3 - lr * grads[5]

        #更新完成

        if step %100 == 0:
            print(epoch,step,'loss:',float(loss))


