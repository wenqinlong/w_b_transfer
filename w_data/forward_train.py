import forward_network as fn
import tensorflow as tf
import pandas as pd
import myutils as ms
import tfrecord

# Hyperparameters
BATCH_SIZE = 128
EPOCH = 5000
LR = 1e-5
cwd = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/w_data/'

# input
x_para = tf.placeholder(tf.float32, shape=[BATCH_SIZE,5], name='structure_parameter')

parameter, index, spectra = tfrecord.read_tfrecord(cwd + '/train_data.tfrecord', BATCH_SIZE)

test_dataset = tfrecord.read_tfrecord(cwd + 'train_data.tfrecord', BATCH_SIZE)

# loss
r_spec = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 603], name='r_spec')
p_spec = fn.fw_net(x_para)
loss = ms.huber_loss(r_spec, p_spec)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-8)
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fw_net')

# minimize
fw_op = optimizer.minimize(loss=loss, var_list=var_list)

# start training
init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    loss_ = []
    epoch = 0
    it_ = []

    for it in range(EPOCH*(30513//BATCH_SIZE)):
        pa, id, sp = sess.run([parameter, index, spectra])
        _, l = sess.run([fw_op, loss], feed_dict={x_para: pa, r_spec: sp})  # _, loss = sess.run([fw_op, loss], feed_dict={x_para: pa, r_spec: sp})
        loss_.append(l)                                                     # TypeError: Fetch argument 0.022959469 has invalid type <class 'numpy.float32'>, must be a string or Tensor. (Can not convert a float32 into a Tensor or Operation.)

        it_.append(it)

        if (it+1) % 238 == 0:
            epoch += 1
            print('Epoch_{}, Iteration_{}, loss: {}'.format(epoch, it, l))
