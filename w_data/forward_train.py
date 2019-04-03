import forward_network as fn
import tensorflow as tf
import pandas as pd
import myutils as ms
import tfrecord
import numpy as np
import os

# Hyperparameters
BATCH_SIZE = 128
EPOCH = 1
LR = 1e-5
cwd = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/w_data/'

# input
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE,5], name='structure_parameter')

para_train, id_train, sp_train = tfrecord.read_tfrecord(cwd + '/train_data.tfrecord', BATCH_SIZE)
para_test, id_test, sp_test = tfrecord.read_tfrecord(cwd + '/test_data.tfrecord', BATCH_SIZE)

test_dataset = tfrecord.read_tfrecord(cwd + 'train_data.tfrecord', BATCH_SIZE)

# loss
r_spec = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 603], name='r_spec')
p_spec = fn.fw_net(x)
loss = ms.huber_loss(r_spec, p_spec)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-8)

# minimize
fw_op = optimizer.minimize(loss=loss)

# start training
init = tf.global_variables_initializer()

train_summary = tf.summary.scalar('Train loss', loss)
test_summary = tf.summary.scalar('Test loss', loss)
# merged_summary = tf.summary.merge_all()  # if want the train and test loss in the same image, then uncomment this line

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session(config=config) as sess:
    sess.run(init)

    # Save model
    export_dir = './savedmodel'
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(p_spec)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x_input': tensor_info_x},
            outputs={'y_output': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )

    builder.add_meta_graph_and_variables(
        sess, tf.saved_model.tag_constants.SERVING,
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature
        }
    )

    builder.save()

    # summary tensorboard
    train_writer = tf.summary.FileWriter('./graphs/train', tf.get_default_graph())
    test_writer = tf.summary.FileWriter('./graphs/test')

    # start train
    epoch = 0
    for it in range(EPOCH*(30513//BATCH_SIZE)):
        pa_tr, id_tr, sp_tr = sess.run([para_train, id_train, sp_train])
        summary_tr, _, l = sess.run([train_summary, fw_op, loss], feed_dict={x: pa_tr, r_spec: sp_tr})  # _, loss = sess.run([fw_op, loss], feed_dict={x_para: pa, r_spec: sp})
        train_writer.add_summary(summary_tr, it)

        if (it+1) % 238 == 0:
            epoch += 1
            print('Epoch_{}, Iteration_{}, loss: {}'.format(epoch, it, l))

    # create dirs and store the test results
    os.makedirs('./test_results/real', exist_ok=True)       # there is no '/' in the last dir
    os.makedirs('./test_results/pred', exist_ok=True)
    real_data = np.empty((0, 609))                          # create a empty ndarray shape = (0, ?)
    pred_data = np.empty((0, 609))

    for i in range(5000//BATCH_SIZE):
        pa_te, id_te, sp_te = sess.run([para_test, id_test, sp_test])

        p_spec_test, summary_te, l_test = sess.run([p_spec, test_summary, loss],
                                                   feed_dict={x: pa_te, r_spec: sp_te})
        test_writer.add_summary(summary_te, i)

        real_results = np.concatenate([id_te, pa_te, sp_te], axis=1)    # (128, 609)
        pred_results = np.concatenate([id_te, pa_te, p_spec_test], axis=1)

        real_data = np.concatenate([real_data, real_results], axis=0)
        pred_data = np.concatenate([pred_data, pred_results], axis=0)

    np.savetxt('test_results/real/real_data.csv', real_data, delimiter=',')  # don't forget about the delimiter=','
    np.savetxt('test_results/pred/pred_data.csv', pred_data, delimiter=',')

train_writer.close()
test_writer.close()

# tensorboard --logdir='/home/qinlong/PycharmProjects/NEU/w_b_transfer/w_data' --port=8008
# Don't enter the graphs folder, don't add spaces before and after '='
