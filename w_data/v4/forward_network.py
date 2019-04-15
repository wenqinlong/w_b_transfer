import myutils as ms
import tensorflow as tf


def fw_net(x, training=True):

    with tf.name_scope('fw_net'):

        with tf.variable_scope('fc1'):
            x = ms.fc(x, 5*256)
            # x = ms.bn(x)
            x = ms.activation(x, relu=True)
            x = tf.reshape(x, [-1, 5, 256])

        with tf.variable_scope('identical_conv1'):

            x = ms.conv1d(x, 256, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 256, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 256, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 256, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)
            print(x.shape)   # (128, 5, 256)

        with tf.variable_scope('t_conv1'):
            x = ms.t_conv1d(x, tf.Variable(tf.random.normal([3, 128, 256])), (128, 10, 128), 2)
            # x = ms.bn(x)
            x = ms.activation(x, relu=True)

        with tf.variable_scope('identical_conv2'):
            x = ms.conv1d(x, 128, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 128, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 128, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 128, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

        with tf.variable_scope('t_conv2'):
            x = ms.t_conv1d(x, tf.Variable(tf.random.normal((3, 64, 128))), (128, 20, 64), 2)
            # x = ms.bn(x)
            x = ms.activation(x, relu=True)

        with tf.variable_scope('identical_conv3'):
            x = ms.conv1d(x, 64, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 64, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 64, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 64, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

        with tf.variable_scope('t_conv3'):
            x = ms.t_conv1d(x, tf.Variable(tf.random.normal((3, 32, 64))), (128, 40, 32), 2)
            # x = ms.bn(x)
            x = ms.activation(x, relu=True)

        with tf.variable_scope('identical_conv4'):
            x = ms.conv1d(x, 32, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 32, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 32, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv1d(x, 32, 3)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

        x = tf.reshape(x, [-1, 40*32])

        with tf.variable_scope('fc2'):
            x = ms.fc(x, units=603)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=False)

    return x
