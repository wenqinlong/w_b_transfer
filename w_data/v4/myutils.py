import tensorflow as tf


def fc(x, units):
    x = tf.layers.dense(inputs=x, units=units)
    return x


def conv1d(x, f, w):
    x = tf.layers.conv1d(inputs=x, filters=f, kernel_size=w, strides=1, padding='same')
    return x


def t_conv1d(x, f, ops, s):
    """
    input: A 3-D `Tensor` of type `float` and shape
      `[batch, in_width, in_channels]` for `NWC` data format or
      `[batch, in_channels, in_width]` for `NCW` data format.
    filters: A 3-D `Tensor` with the same type as `value` and shape
      `[filter_width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor`, containing three elements, representing the
      output shape of the deconvolution op.
    strides: An int or list of `ints` that has length `1` or `3`.  The number of
      entries by which the filter is moved right at each step.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. `'NWC'` and `'NCW'` are supported.
    dilations: An int or list of `ints` that has length `1` or `3` which
      defaults to 1. The dilation factor for each dimension of input. If set to
      k > 1, there will be k-1 skipped cells between each filter element on that
      dimension. Dilations in the batch and depth dimensions must be 1.
    name: Optional name for the returned tensor.
    Returns:
    A `Tensor` with the same type as `value`.
    """
    x = tf.contrib.nn.conv1d_transpose(value=x, filter=f, output_shape=ops, stride=2, padding='SAME')
    return x


def conv2d(x, f, w):
    x = tf.layers.conv2d(inputs=x, filters=f, kernel_size=w, strides=1, padding='same')
    return x


def t_conv2d(x, f, w, s):
    x = tf.layers.conv2d_transpose(x, filters=f, kernel_size=w, strides=s, padding='same')
    return x


def max_pool(x, ps, s):
    x = tf.layers.max_pooling2d(inputs=x, pool_size=ps, stides=s)
    return x


def bn(x, training=True):
    x = tf.layers.batch_normalization(inputs=x, axis=-1, momentum=0.99, epsilon=1e-3, training=training)
    return x


def activation(x, relu=True):
    if relu:
        x = tf.nn.relu(x, name='relu')
    else:
        x = tf.nn.tanh(x)
    return x


def data_norm(x, para, reverse=False):
    if para == 'top_length':
        max = 1600
        min = 1000
    elif para == 'bottom_length':
        max = 1600
        min = 1000
    elif para == 'top_spacer':
        max = 600
        min = 200
    elif para == 'bottom_spacer':
        max = 600
        min = 200
    elif para == 'angle':
        max = 180
        min = 0
    else:
        max = 1
        min = 0

    if not reverse:
        x = 2.0 * (x - min) / (max - min) - 1   # confine the data to range [-1, 1]
    else:
        x = 0.5 * (x + 1) * (max - min) + min
    return x


def mse_loss(y, y_):
    loss = tf.losses.mean_squared_error(y, y_)
    return loss


def huber_loss(y, y_, delta=0.1):
    loss = tf.losses.huber_loss(y_, y, delta)
    return loss
