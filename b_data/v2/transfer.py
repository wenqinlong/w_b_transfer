import tensorflow as tf
import tfrecord as tfr
import myutils as ms
import os
import numpy as np

# Hyperparameter
BATCH_SIZE = 128
LR = 1E-5
EPOCH = 2000
CWD = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/'

# Read data
test_para, test_rp = tfr.read_tfrecord(CWD + 'b_data/test_2000.tfrecord', BATCH_SIZE)
train_para, train_rp = tfr.read_tfrecord(CWD + 'b_data/train_6393.tfrecord', BATCH_SIZE)

graph = tf.Graph()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
x_key = 'x_input'
# y_key = 'y_output'

export_dir = CWD + 'w_data/v2/savedmodel_1'

y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 402], name='simu')

with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING],
        export_dir
    )

    signature = meta_graph_def.signature_def
    x_tensor_name = signature[signature_key].inputs[x_key].name
    # y_tensor_name = signature[signature_key].inputs[y_key].name

    x = sess.graph.get_tensor_by_name(x_tensor_name)
    # y = sess.graph.get_tensor_by_name(y_tensor_name)

    connect = sess.graph.get_operation_by_name('fw_net/identical_conv3/relu_3').outputs[0]

    with tf.name_scope('trans_part'):
        with tf.variable_scope('trans_part'):
            with tf.variable_scope('t_conv3'):
                t = ms.t_conv2d(connect, 32, [2, 2], 2)
                # x = ms.bn(x)
                t = ms.activation(t, relu=True)

            with tf.variable_scope('identical_conv4'):
                t = ms.conv(t, 32, [3, 3])
                # x = ms.bn(x, training=training)
                t = ms.activation(t, relu=True)

                t = ms.conv(t, 32, [3, 3])
                # x = ms.bn(x, training=training)
                t = ms.activation(t, relu=True)

                t = ms.conv(t, 32, [3, 3])
                # x = ms.bn(x, training=training)
                t = ms.activation(t, relu=True)

                t = ms.conv(t, 32, [3, 3])
                # x = ms.bn(x, training=training)
                t = ms.activation(t, relu=True)

            t = tf.reshape(t, [-1, 40 * 40 * 32])

            with tf.variable_scope('fc2'):
                t = ms.fc(t, units=402)
                # x = ms.bn(x, training=training)
                t = ms.activation(t, relu=False)

    new_loss = ms.huber_loss(t, y_)
    new_optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1e-8,
                                           name='new_adam')

    new_op = new_optimizer.minimize(loss=new_loss)

    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='trans_part')

    sess.run(tf.variables_initializer(var_list=var))
    sess.run(tf.variables_initializer(new_optimizer.variables()))

    # summary tensorboard
    train_summary = tf.summary.scalar('Train loss', new_loss)
    test_summary = tf.summary.scalar('Test loss', new_loss)
    train_writer = tf.summary.FileWriter('./graphs/train', tf.get_default_graph())
    test_writer = tf.summary.FileWriter('./graphs/test')

    # start train
    epoch = 0
    for it in range(EPOCH * (6393 // BATCH_SIZE)):
        tr_para, tr_rp = sess.run([train_para, train_rp])
        summary_tr, _, l = sess.run([train_summary, new_op, new_loss],
                                    feed_dict={x: tr_para, y_: tr_rp})  # _, loss = sess.run([fw_op, loss], feed_dict={x_para: pa, r_spec: sp})
        train_writer.add_summary(summary_tr, it)

        if (it + 1) % 49 == 0:
            epoch += 1

        # create dirs and store the test results
        if ((it + 1) % (49 * 500)) == 0:
            print('Epoch_{}, Iteration_{}, loss: {}'.format(epoch, it, l))
            os.makedirs('./test_results/epoch_{}'.format(epoch), exist_ok=True)  # there is no '/' in the last dir
            os.makedirs('./test_results/epoch_{}'.format(epoch), exist_ok=True)
            simu_data = np.empty((0, 407))                     # create a empty ndarray shape = (0, ?)
            pred_data = np.empty((0, 407))

            for i in range(2000 // BATCH_SIZE):
                te_para, te_rp = sess.run([test_para, test_rp])

                p_te_rp, summary_te, l_test = sess.run([t, test_summary, new_loss],
                                                       feed_dict={x: te_para, y_: te_rp})
                test_writer.add_summary(summary_te, i)

                simu_results = np.concatenate([te_para, te_rp], axis=1)  # (128, 407)
                pred_results = np.concatenate([te_para, p_te_rp], axis=1)

                simu_data = np.concatenate([simu_data, simu_results], axis=0)
                pred_data = np.concatenate([pred_data, pred_results], axis=0)

            np.savetxt('test_results/epoch_{}/simu_data.csv'.format(epoch), simu_data, delimiter=',')  # don't forget about the delimiter=','
            np.savetxt('test_results/epoch_{}/pred_data.csv'.format(epoch), pred_data, delimiter=',')

train_writer.close()
test_writer.close()
