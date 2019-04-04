import tensorflow as tf
import pandas as pd
import numpy as np
from random import sample

def split_dataset(x, test_num, total):
    '''
    :param x: dataset
    :param test_num: the nunber of test data
    :param total: the total number of dataset
    :return: two dataset for test and train
    '''
    test_loc = sample(range(total), test_num)
    x_test = x.iloc[test_loc, :]
    x_train = x.drop(test_loc, axis=0)
    return x_test, x_train



def tfrecord(x, num, y):
    '''
    :param x: dataset need to be converted tfrecord
    :param num: the number of data
    :param y: tfrecords file
    :return: None
    '''
    writer = tf.python_io.TFRecordWriter(y)
    for i in range(num):
        para = x.iloc[i, 1:5]        # 4
        response = x.iloc[i, 5:407]  # 404  optical response

        example = tf.train.Example(features=tf.train.Features(feature={
            'para': tf.train.Feature(float_list=tf.train.FloatList(value=[i for i in para])),
            'response': tf.train.Feature(float_list=tf.train.FloatList(value=[j for j in response]))
        }))
    writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord(tfrecord, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord)

    def parse(record):
        features = {
            'para': tf.FixedLenFeature([5], tf.float32),
            'response': tf.FixedLenFeature([402], tf.float32)
        }
        parsed = tf.parse_single_example(record, features)
        para = parsed['para']
        response = parsed['response']

        return para, response

    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=10000)        # better >= the number of data
    dataset = dataset.prefetch(buffer_size=batch_size)  # need to confirm
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()

    return data





if __name__ == '__main__':
    data = pd.read_csv('./data/raw_data/data_norm_padding_406.csv')
    test, train = split_dataset(data,2000)
    tfrecord(test, 2000, './test_2000.tfrecord')
    tfrecord(train, 6393, './train_6393.tfrecord')