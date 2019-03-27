import forward_network as fn
import tensorflow as tf
import pandas as pd
import myutils as ms
import tfrecord

# Hyperparameters
BATCH_SIZE = 64
EPOCH = 1
LR = 1e-5


# input
strupata = tf.placeholder(tf.float32, shape=[BATCH_SIZE,5], name='structure_parameter')

train_dataset = tfrecord.read_tfrecord('E:/Tensorflow/wei_data_practice/train_data.tfrecord', BATCH_SIZE, EPOCH)
train_iterator = train_dataset.make_one_shot_iterator()
train_next = train_iterator.get_next()

test_dataset = tfrecord.read_tfrecord('E:/Tensorflow/wei_data_practice/train_data.tfrecord', BATCH_SIZE, EPOCH)
test_iterator = test_dataset.make_one_shot_iterator()
test_next = test_iterator.get_next()

# loss
r_spec = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 603], name='r_spec')
p_spec = fn.fw_net(strupata)
mse_loss = ms.mse_loss(r_spec, p_spec)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-8)
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fw_net')

# minimize
fw_op = optimizer.minimize(loss=mse_loss, var_list=var_list)

# start training
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    loss_ = []
    epoch_ = []
    it_ = []
    for epoch in range(EPOCH + 1):
        for it in range(30513//64):

            para_batch, label_batch, spec_batch = sess.run(next)
            _, loss = sess.run([fw_op, mse_loss], feed_dict={strupata: para_batch, r_spec: spec_batch})
            loss_.append(loss)
            it_.append(it)

            if it % 10 == 0:
                print('Epoch_{}, Iteration_{}, loss: {}'.format(epoch, it, loss))




        # test


