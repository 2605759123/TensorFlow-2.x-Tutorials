# coding=gbk
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# �������ݼ�
(x, y), (x_val, y_val) = datasets.mnist.load_data()
# x����60K��28��28�� y����60K��
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.#ת��Ϊtensor�ĸ�ʽ
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)#һ�η���200��ͼƬһ�����

# for step,(x,y) in enumerate(train_dataset):
#     print(step,x.shape,y,y.shape)



model = keras.Sequential([
    layers.Dense(512, activation='relu'),#512ָ�������512
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)#����ݶ��½�


def train_epoch(epoch):

    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):


        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28*28))
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())



def train():

    for epoch in range(30):

        train_epoch(epoch)






if __name__ == '__main__':
    train()