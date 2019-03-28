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

para_train, id_train, sp_train = tfrecord.read_tfrecord(cwd + '/train_data.tfrecord', BATCH_SIZE)
para_test, id_test, sp_test = tfrecord.read_tfrecord(cwd + '/test_data.tfrecord', BATCH_SIZE)

test_dataset = tfrecord.read_tfrecord(cwd + 'train_data.tfrecord', BATCH_SIZE)

# loss
r_spec = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 603], name='r_spec')
p_spec = fn.fw_net(x_para)
train_loss = ms.huber_loss(r_spec, p_spec)
test_loss = ms.huber_loss(r_spec, p_spec)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-8)

# minimize
fw_op = optimizer.minimize(loss=train_loss)

# start training
init = tf.global_variables_initializer()

tf.summary.scalar('Train loss', train_loss)
tf.summary.scalar('Test loss', test_loss)
merged_summary = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session(config=config) as sess:
    sess.run(init)

    writer = tf.summary.FileWriter('./gpaphs', tf.get_default_graph())
    writer.add_graph(sess.graph)

    epoch = 0

    for it in range(EPOCH*(30513//BATCH_SIZE)):
        pa_tr, id_tr, sp_tr = sess.run([para_train, id_train, sp_train])

        summary_tr, _, l = sess.run([merged_summary, fw_op, train_loss], feed_dict={x_para: pa_tr, r_spec: sp_tr})  # _, loss = sess.run([fw_op, loss], feed_dict={x_para: pa, r_spec: sp})
        writer.add_summary(summary_tr, it)

        if (it+1) % 238 == 0:
            epoch += 1
            print('Epoch_{}, Iteration_{}, loss: {}'.format(epoch, it, l))

    for i in range(5000//64):
        pa_te, id_te, sp_te = sess.run([para_test, id_test, sp_test])
        summary_te, l_test = sess.run([merged_summary, test_loss], feed_dict={x_para: pa_te, r_spec: sp_te})
        writer.add_summary(summary_te, i)

        if (i+1) % 100 == 0:
            print('Iteration_{}, loss: {}'.format(i, l_test))

writer.close()

# tensorboard --logdir='/home/qinlong/PycharmProjects/NEU/w_b_transfer/w_data' --port=8008
# Don't enter the graphs folder, don't add spaces before and after '='
