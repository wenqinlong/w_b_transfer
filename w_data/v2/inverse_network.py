import tensorflow as tf
import myutils as ms
import forward_network as fn

def iv_net(x, training=True):
    with tf.variable_scope('iv_net', reuse=tf.AUTO_REUSE):
        x = ms.fc(x, units=40*40*32)
        x = ms.bn(x, training=training)
        x = tf.reshape(x, [-1, 40, 40, 32])
        x = fn.id_blk(x, 32, [3, 3])
        x = fn.id_blk(x, 32, [3, 3])
        x = ms.max_pool(x, [2, 2], 2)

        x = ms.conv(x, 64, [3, 3])
        x = ms.bn(x, training=training)
        x = ms.activation(x, relu=True)
        x = fn.id_blk(x, 64, [3, 3])
        x = fn.id_blk(x, 64, [3, 3])
        x = ms.max_pool(x, [2, 2], 2)

        x = ms.conv(x, 128, [3, 3])
        x = ms.bn(x, training=training)
        x = ms.activation(x, relu=True)
        x = fn.id_blk(x, 128, [3, 3])
        x = fn.id_blk(x, 128, [3, 3])
        x = ms.max_pool(x, [2, 2], 2)

        x = ms.conv(x, 256, [3, 3])
        x = ms.bn(x, training=training)
        x = ms.activation(x, relu=True)
        x = fn.id_blk(x, 256, [3, 3])
        x = fn.id_blk(x, 256, [3, 3])

        x = tf.reshape(x, [-1, 5*5*256])
        x = ms.fc(x, units=5)
        x = ms.activation(x, relu=False)

    return x

