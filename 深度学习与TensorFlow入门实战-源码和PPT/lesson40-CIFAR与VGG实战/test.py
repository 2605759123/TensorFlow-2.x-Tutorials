import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os

#添加随机数种子
tf.random.set_seed(2345)

#设置卷积层网络
conv_layers = [#5units of conv(convolution) + max pooling
    #unit 1
    #64:输出的层数 w:[3,3]（滑动窗口） N设置为64，输出的C层增加为64（无论输入是多少）作用：剔除位置信息，引入更高层的信息概念
    layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    #剔除位置信息，位置信息参数量减半
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    #unit 2
    #64:输出的层数 w:[3,3]（滑动窗口） N设置为128，输出的C层增加为128（无论输入是多少,实际上是上层的64）
    layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    #unit 3
    #64:输出的层数 w:[3,3]（滑动窗口） N设置为256，输出的C层增加为256（无论输入是多少）
    layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    #unit 4
    #64:输出的层数 w:[3,3]（滑动窗口） N设置为512，输出的C层增加为512（无论输入是多少）
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    #unit 5
    #64:输出的层数 w:[3,3]（滑动窗口） N设置为128，输出的C层保持不变，为了不使参数量过多
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
    # [b,-1]=> [b,1,1,512]
]

#进行预处理
def preprocess(x,y):
    x= tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y

(x,y),(x_test,y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y,axis=1)
#把1去掉 y：[b,1]=> [b]
y_test = tf.squeeze(y_test,axis=1)
print(x.shape,y.shape,x_test.shape,y_test.shape)

train_db=tf.data.Dataset.from_tensor_slices((x,y))
train_db=train_db.shuffle(1000).map(preprocess).batch(64)

test_db=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db=test_db.shuffle(1000).map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample:',sample[0].shape,sample[1].shape,
      tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))


def main():
    conv_net = Sequential(conv_layers)
    # conv_net.build(input_shape=[None,32,32,3])
    # x = tf.random.normal([4,32,32,3])
    # out = conv_net(x)
    # print(out.shape)


    fc_net = Sequential([
        #设置全连接层网络 Fully connected layer
        layers.Dense(256,activation=tf.nn.relu),
        layers.Dense(128,activation=tf.nn.relu),
        layers.Dense(100,activation=tf.nn.relu)
    ])

    conv_net.build(input_shape=[None,32,32,3])
    fc_net.build(input_shape=[None,512])
    optimizer = optimizers.Adam(lr=1e-4)

    #可求导参数
    # +:[1,2]+[3,4] => [1,2,3,4]
    variables = conv_net.trainable_variables + fc_net.trainable_variables
    # print(conv_net.trainable_variables)

    for epoch in range(50):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [b,32,32,3] => [b,1,1,512] 卷积层 convolution
                out = conv_net(x)
                #flatten(扁平化) , => [b,512]
                out = tf.reshape(out,[-1,512])
                # [b,512] => [b,100] 全连接层 fully connected layer
                logits = fc_net(out)
                # [b] => [b,100]
                y_onehot = tf.one_hot(y,depth=100)
                # computer loss
                loss = tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss,variables)
            optimizer.apply_gradients(zip(grads,variables))

            if step % 10 ==0:
                print(epoch,step,'loss:',float(loss))

        total_num=0
        total_correct=0
        for x,y in test_db:
            out = conv_net(x)
            out = tf.reshape(out,[-1,512])
            logits = fc_net(out)
            prob=tf.nn.softmax(logits,axis=1)
            pred = tf.nn.softmax(logits,axis=1)
            pred.tf.cast(pred,dtype=tf.int32)

            correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch,'acc:',acc)




if __name__ == '__main__':
    main()

