import forward_network as fn
import tensorflow as tf
import pandas as pd
import myutils as ms
import tfrecord

# Hyperparameters
BATCH_SIZE = 128
EPOCH = 2
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
tf.summary.scalar('loss', loss)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-8)

# var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fw_net')

# minimize
fw_op = optimizer.minimize(loss=loss)

# start training
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session(config=config) as sess:
    sess.run(init)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./gpaphs', tf.get_default_graph())
    writer.add_graph(sess.graph)

    epoch = 0

    for it in range(EPOCH*(30513//BATCH_SIZE)):
        pa, id, sp = sess.run([parameter, index, spectra])

        s = sess.run(merged_summary, feed_dict={x_para: pa, r_spec: sp})
        writer.add_summary(s, it)

        _, l = sess.run([fw_op, loss], feed_dict={x_para: pa, r_spec: sp})  # _, loss = sess.run([fw_op, loss], feed_dict={x_para: pa, r_spec: sp})

        if (it+1) % 238 == 0:
            epoch += 1
            print('Epoch_{}, Iteration_{}, loss: {}'.format(epoch, it, l))


writer.close()

# tensorboard --logdir='/home/qinlong/PycharmProjects/NEU/w_b_transfer/w_data' --port=8008
# Don't enter the graphs folder, don't add spaces before and after '='
