import myutils as ms
import tensorflow as tf


def fw_net(x, training=True):

    with tf.name_scope('fw_net'):

        with tf.variable_scope('fc1'):
            x = ms.fc(x, 5*5*256)
            # x = ms.bn(x)
            x = ms.activation(x, relu=True)
            x = tf.reshape(x, [-1, 5, 5, 256])

        with tf.variable_scope('identical_conv1'):

            x = ms.conv(x, 256, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 256, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 256, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 256, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            # x = ms.conv(x, 256, [3, 3])
            # # x = ms.bn(x, training=training)
            # x = ms.activation(x, relu=True)

        with tf.variable_scope('t_conv1'):
            x = ms.t_conv2d(x, 128, [2, 2], 2)
            # x = ms.bn(x)
            x = ms.activation(x, relu=True)

        with tf.variable_scope('identical_conv2'):
            x = ms.conv(x, 128, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 128, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 128, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 128, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)
            #
            # x = ms.conv(x, 128, [3, 3])
            # # x = ms.bn(x, training=training)
            # x = ms.activation(x, relu=True)

        with tf.variable_scope('t_conv2'):
            x = ms.t_conv2d(x, 64, [2, 2], 2)
            # x = ms.bn(x)
            x = ms.activation(x, relu=True)

        with tf.variable_scope('identical_conv3'):
            x = ms.conv(x, 64, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 64, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 64, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 64, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            # x = ms.conv(x, 64, [3, 3])
            # # x = ms.bn(x, training=training)
            # x = ms.activation(x, relu=True)

        with tf.variable_scope('t_conv3'):
            x = ms.t_conv2d(x, 32, [2, 2], 2)
            # x = ms.bn(x)
            x = ms.activation(x, relu=True)

        with tf.variable_scope('identical_conv4'):
            x = ms.conv(x, 32, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 32, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 32, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            x = ms.conv(x, 32, [3, 3])
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=True)

            # x = ms.conv(x, 32, [3, 3])
            # # x = ms.bn(x, training=training)
            # x = ms.activation(x, relu=True)

        x = tf.reshape(x, [-1, 40*40*32])

        with tf.variable_scope('fc2'):
            x = ms.fc(x, units=603)
            # x = ms.bn(x, training=training)
            x = ms.activation(x, relu=False)

    return x
