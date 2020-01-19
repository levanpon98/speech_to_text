from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTM,
    "gru": tf.keras.layers.GRU,
}

# Parameters for batch normalization.
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997

# Filters of convolution layer
_CONV_FILTERS = 32


def batch_norm(input, training):
    """
    Batch normalization layer.
    :param input: input data for batch norm layer
    :param training: a boolean to indicate if it is in training stage
    :return: a tensor output batch norm layer
    """

    return tf.keras.layers.BatchNormalization(
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        trainable=training
    )(input)


def conv_bn_layer(inputs, padding, filters, kernel_size, strides, layer_id, training):
    """
    2D convolutional + batch normalization layer

    :param inputs: input data for convolution layer
    :param padding: padding to be applied before convolution layer
    :param filters: an integer, number of output filters int the convolution
    :param kernel_size: a tuple specifying the height and width of the 2D convolution window
    :param strides: a tuple specifying the stride length of the convolution
    :param layer_id: an integer specifying the layer index
    :param training: a boolean to indicate which stage we are in (training/eval)
    :return: tensor output from the current layer
    """
    inputs = tf.pad(
        inputs,
        [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]]
    )

    inputs = tf.keras.layers.Conv2D(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding='valid',
                                    use_bias=False,
                                    activation=tf.nn.relu6,
                                    name="cnn_{}".format(layer_id))(inputs)
    return batch_norm(inputs, training)


def _rnn_layer(inputs, rnn_cell, rnn_hidden_size, layer_id, is_batch_norm,
               is_bidirectional, training):
    """
    Defines a batch normalization + rnn layer.
    :param inputs: input tensors for the current layer.
    :param rnn_cell: RNN cell instance to use.
    :param rnn_hidden_size: an integer for the dimensionality of the rnn output space.
    :param layer_id: an integer for the index of current layer
    :param is_batch_norm: a boolean specifying whether to perform batch normalization on input states.
    :param is_bidirectional: a boolean specifying whether the rnn layer is bi-directional.
    :param training: a boolean to indicate which stage we are in (training/eval).
    :return: tensor output for the current layer.
    """
    if is_batch_norm:
        inputs = batch_norm(inputs, training)

    # Construct forward/backward RNN cells.
    fw_cell = rnn_cell(units=rnn_hidden_size, go_backwards=True, return_sequences=True,trainable=training,
                       name="rnn_fw_{}".format(layer_id))
    bw_cell = rnn_cell(units=rnn_hidden_size, return_sequences=True, trainable=training,
                       name="rnn_bw_{}".format(layer_id))

    if is_bidirectional:
        rnn_outputs = tf.keras.layers.Bidirectional(fw_cell, merge_mode='concat', backward_layer=bw_cell)(inputs)
        # rnn_outputs = tf.concat(rnn_outputs, -1)
    else:
        rnn_outputs = fw_cell(inputs)

    return rnn_outputs


class DeepSpeech2(tf.keras.Model):
    """Define DeepSpeech2 model."""

    def __init__(self, num_rnn_layers, rnn_type, is_bidirectional,
                 rnn_hidden_size, num_classes, use_bias, **kwargs):
        """Initialize DeepSpeech2 model.

        Args:
          num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
          rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
          is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
          rnn_hidden_size: an integer for the number of hidden states in each unit.
          num_classes: an integer, the number of output classes/labels.
          use_bias: a boolean specifying whether to use bias in the last fc layer.
        """
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.use_bias = use_bias
        super(DeepSpeech2, self).__init__(**kwargs)

    def call(self, inputs, training=None, mask=None):
        # Two cnn layers.
        inputs = conv_bn_layer(
            inputs, padding=(20, 5), filters=_CONV_FILTERS, kernel_size=(41, 11),
            strides=(2, 2), layer_id=1, training=training)

        inputs = conv_bn_layer(
            inputs, padding=(10, 5), filters=_CONV_FILTERS, kernel_size=(21, 11),
            strides=(2, 1), layer_id=2, training=training)

        # output of conv_layer2 with the shape of
        # [batch_size (N), times (T), features (F), channels (C)].
        # Convert the conv output to rnn input.
        batch_size = tf.shape(inputs)[0]
        feat_size = inputs.get_shape().as_list()[2]
        inputs = tf.reshape(
            inputs,
            [batch_size, -1, feat_size * _CONV_FILTERS])

        # RNN layers.
        rnn_cell = SUPPORTED_RNNS[self.rnn_type]
        for layer_counter in xrange(self.num_rnn_layers):
            # No batch normalization on the first layer.
            is_batch_norm = (layer_counter != 0)
            inputs = _rnn_layer(
                inputs, rnn_cell, self.rnn_hidden_size, layer_counter + 1,
                is_batch_norm, self.is_bidirectional, training)

        # FC layer with batch norm.
        inputs = batch_norm(inputs, training)
        logits = tf.keras.layers.Dense(units=self.num_classes, use_bias=self.use_bias, trainable=training)(inputs)

        return logits