import tensorflow as tf
import numpy as np
import inverse_network as iv
import pandas as pd
import tfrecord
import myutils as ms
from time import time

# Hyperparameters
EPOCH = 1
BATCH_SIZE = 128
LR = 1e-3
Training = True

# placeholder
x = tf.placeholder(dtype=tf.float32, shape=[None, 603])
y = tf.placeholder(dtype=tf.float32, shape=[None, 5])

# loss
y_ = iv.iv_net(x, training=Training)
loss = ms.mse_loss(y, y_)

optimizer = tf.train.AdamOptimizer(learning_rate=LR,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-8)
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='iv_net')
iv_op = optimizer.minimize(loss=loss, var_list=var_list)

init = tf.global_variables_initializer()

# train data
tr_para, tr_label, tr_spec = tfrecord.read_tfrecord('./train_data.tfrecord', BATCH_SIZE)
train_loss_ = []
epoch_ = []
tr_it_ = []

# test_data
te_para, te_label, te_spec = tfrecord.read_tfrecord('./test_data.tfrecord', BATCH_SIZE)
test_loss_ = []
te_it_ = []

saver = tf.train.Saver()

# configure gpu
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)
with tf.Session(config=config) as sess:
    if Training:
        start_train = time()
        sess.run(init)
        epoch = 0

        for it in range(EPOCH*25513//BATCH_SIZE):
            tr_pa, tr_la, tr_sp = sess.run([tr_para, tr_label, tr_spec])
            train_loss, _ = sess.run([loss, iv_op], feed_dict={x: tr_sp, y: tr_pa})
            train_loss_.append(train_loss)
            tr_it_.append(it)

            if it % 10 == 0:
                print('Epoch_{}, Iteration_{}, train_loss: {}'.format(epoch, it, train_loss))

            if (it + 1) % 199 == 0:
                epoch += 1

            if (epoch + 1) % 1 == 0:
                save_path = saver.save(sess, './iv_checkpoint/EPOCH_{}_iv.ckpt'.format(epoch))

        df_trian_loss = pd.DataFrame({'Iteration': tr_it_,
                                      'Train_loss': train_loss_})
        df_trian_loss.to_csv('./iv_train_loss_{}.csv'.format(EPOCH))
        end_train = time()
        print('It took {} seconds to train iv_net'.format(end_train-start_train))

    else:
        start_test = time()
        model_path = './iv_checkpoint/EPOCH_{}_iv.ckpt'.format()
        meta_path = './iv_checkpoint/EPOCH_{}_iv.ckpt.meta'.format()
        saver_ = tf.train.import_meta_graph(meta_path)
        saver_.restore(model_path)

        for i in range(5000//BATCH_SIZE):
            te_pa, te_la, te_sp = sess.run([te_para, te_label, te_para])
            te_pa_pred, test_loss = sess.run([y_, loss], feed_dict={x: te_sp, y: te_pa})
            pred_data = np.hstack((te_la, te_pa_pred, te_sp))
            test_loss_.append(test_loss)
            test_data = np.hstack((te_la, te_pa, te_sp))

            if i == 0:
                test_data_ = test_data
                pred_data_ = pred_data

            else:
                test_data_ = np.vstack((test_data_, test_data))
                pred_data_ = np.vstack((pred_data_, pred_data))

            te_it_.append(i)
            df_test_loss = pd.DataFrame({'Iteration': te_it_,
                                         'Test_loss': test_loss_})
            df_test_loss.to_csv('./iv_test_results/Epoch_{}_test'.format(epoch))

            np.savetxt('./iv_test_results/Epoch_{}_iv_test_data.csv',format(epoch), test_data_, delimiter=',')
            np.savetxt('./iv_test_results/Epoch_{}_iv_pred_data.csv', format(epoch), pred_data_, delimiter=',')
            end_test = time()
            print('It took {} seconds to test.'.format(end_test - start_test))

