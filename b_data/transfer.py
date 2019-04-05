import tensorflow as tf
import tfrecord as tfr

# Hyperparameter
BATCH_SIZE = 128
LR = 1E-5
EPOCH = 2
CWD = '/home/qinlong/PycharmProjects/NEU/w_b_transfer/'

# Read data
test_para, test_ref = tfr.read_tfrecord('./test_2000.tfrecord', BATCH_SIZE)
train, train_ref = tfr.read_tfrecord('./train_2000.tfrecord', BATCH_SIZE)

graph = tf.Graph()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
x_key = 'x_input'
y_key = 'y_output'

export_dir = CWD + 'w_data/savemodel'

with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_costants.SERVING],
        export_dir
    )

    signature = meta_graph_def.signature_def
    x_tensor_name = signature[signature_key].inputs[x_key].name
    y_tensor_name = signature[signature_key].inputs[y_key].name

    x = sess.graph.get_tensor_by_name(x_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)

    connect = sess.graph.get_operation_by_name('fw_net/identical_conv3/relu_4').outputs[0]

    with tf.name_scope('trans_part'):


