import os
import tensorflow as tf
import numpy as np
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tfrecord(path, out):
    writer = tf.python_io.TFRecordWriter(out)
    file = []
    for i in os.listdir(path):
        file.append(i)
    file.sort()
    print(file)
    LL_norm = pd.read_csv(path+file[0])
    LR_norm = pd.read_csv(path+file[1])
    RR_norm = pd.read_csv(path+file[2])
    for index in LL_norm.index.values:
        label = LL_norm.loc[index][0]
        para = np.array(LL_norm.loc[index][1:6])
        spec_LL = LL_norm.loc[index][6:]
        spec_LR = LR_norm.loc[index][6:]
        spec_RR = RR_norm.loc[index][6:]
        # print(spec_RR.shape)    # (201,)
        spec = np.concatenate([spec_LL, spec_LR, spec_RR], axis=0)   # (201,), so the axis = 0

        example = tf.train.Example(features=tf.train.Features(feature={                          # tf.train.Example
            'para': tf.train.Feature(float_list=tf.train.FloatList(value=[i for i in para])),    # float
            'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            'spec': tf.train.Feature(float_list=tf.train.FloatList(value=[i for i in spec]))
        }))

        writer.write(example.SerializeToString())  # Don't forget the ()
    writer.close()
    return out


def read_tfrecord(fn, batch_size):
    dataset = tf.data.TFRecordDataset(fn)

    def parser(record):
        features = {
            'para': tf.FixedLenFeature([5], tf.float32),
            'label': tf.FixedLenFeature([1], tf.float32),
            'spec': tf.FixedLenFeature([603], tf.float32)    # 加上具体的维度数据  否则报错 Can't parse serialized Example
        }
        parsed = tf.parse_single_example(record, features)
        para = parsed['para']
        label = parsed['label']
        spec = parsed['spec']

        return para, label, spec

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()

    return data


if __name__ == '__main__':
    CWD = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/w_data/'
    tfrecord(CWD+'data/test_norm/', 'test_data.tfrecord')
    tfrecord(CWD+'data/train_norm/', 'train_data.tfrecord')

    para_batch, label_batch, spec_batch = read_tfrecord(CWD + 'test_data.tfrecord', 2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2):
            p, l, s = sess.run([para_batch, label_batch, spec_batch])
            #para_batch, label_batch, spec_batch = sess.run([para_batch, label_batch, spec_batch])
            print(s.shape)
            print(l)