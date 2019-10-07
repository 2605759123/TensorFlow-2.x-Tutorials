import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics

def preprocess(x,y):
    # x=tf.convert_to_tensor(x,dtype=tf.float32)/255.
    # y=tf.convert_to_tensor(y,dtype=tf.int32)
    x=tf.cast(x,dtype=tf.float32)/255.
    y=tf.cast(y,dtype=tf.int32)
    return x,y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape,y.shape)



db=tf.data.Dataset.from_tensor_slices((x,y))
#shuffle:打乱数据集
batchsz=128
db=db.map(preprocess).shuffle(10000).batch(batchsz)

db_test=tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test=db_test.map(preprocess).batch(batchsz)

#产生迭代器对象
db_iter=iter(db)
sample=next(db_iter)

print(sample[0].shape,sample[1].shape)

model=Sequential([
    #[b,784] => [b,256]
    layers.Dense(256,activation=tf.nn.relu),
    layers.Dense(128,activation=tf.nn.relu),
    layers.Dense(64,activation=tf.nn.relu),
    layers.Dense(32,activation=tf.nn.relu),
    layers.Dense(10)
])

model.build(input_shape=[None,28*28])
model.summary()

#w = w-lr*grads
optimizer = optimizers.Adam(lr=1e-3)


def main():
    for epoch in range(30):
        for step,(x,y) in enumerate(db):
            x=tf.reshape(x,[-1,28*28])
            # x:[b,28,28] => [b,784]
            # y:[b]
            with tf.GradientTape() as tape:
                #[b,784] => [b,10]
                logits=model(x)
                y_onehot=tf.one_hot(y,depth=10)
                # [b]
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot,logits))
                loss_ce=tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True))

            # grads= tape.gradient(loss_ce,model.trainable_variables)
            grads= tape.gradient(loss_mse,model.trainable_variables)

            optimizer.apply_gradients(zip(grads,model.trainable_variables))

            if step % 100 == 0:
                print(epoch,step,'loss:',float(loss_ce),float(loss_mse))
        #test
        total_correct = 0
        total_number = 0
        for x,y in db_test:
            x=tf.reshape(x,[-1,28*28])
            # x:[b,28,28] => [b,784]
            # y:[b]

            # [b,10]
            logits=model(x)
            #logits => prob,[b,10]
            prob = tf.nn.softmax(logits,axis=1)
            # [b,10] => [b]
            pred = tf.argmax(prob,axis=1)
            pred=tf.cast(pred,dtype=tf.int32)
            #pred:[b]
            #y:[b]

            #correct:[b],True:equel False:not equel
            correct = tf.equal(pred,y)
            correct = tf.reduce_sum(tf.cast(correct,dtype=tf.int32))
            total_correct+= int(correct)
            total_number += x.shape[0]

        acc = total_correct/total_number
        print('epoch:',acc)


if __name__ == '__main__':
    main()