import tensorflow as tf
import tensorflow.keras.layers as layers

class ConvNeXt_Block(layers.Layer):
    r""" ConvNeXt Block.
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = layers.DepthwiseConv2D(kernel_size=7, padding='same')  # depthwise conv
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = layers.Dense(4 * dim)
        self.act = layers.Activation('gelu')
        self.pwconv2 = layers.Dense(dim)
        self.drop_path = DropPath(drop_path)
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value

    def build(self, input_shape):
        self.gamma = tf.Variable(
            initial_value=self.layer_scale_init_value * tf.ones((self.dim)),
            trainable=True,
            name='_gamma')
        self.built = True

    def call(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + self.drop_path(x)
        return x

class Downsample_Block(layers.Layer):
    """The Downsample Block in ConvNeXt

        Args:
            dim (int): number of channels
    """

    def __init__(self, dim):
        super().__init__()
        self.LN = layers.LayerNormalization(epsilon=1e-6)
        self.conv = layers.Conv2D(dim, kernel_size=2, strides=2)

    def build(self, input_shape):
        self.built = True

    def call(self, x):
        x = self.LN(x)
        x = self.conv(x)
        return x

class DropPath(tf.keras.layers.Layer):
    """The Drop path in ConvNeXt

        Reference:
            https://github.com/rishigami/Swin-Transformer-TF/blob/main/swintransformer/model.py
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return self._drop_path(x, self.drop_prob, training)

    
    def _drop_path(self, inputs, drop_prob, is_training):
        if (not is_training) or (drop_prob == 0.):
            return inputs

        # Compute keep_prob
        keep_prob = 1.0 - drop_prob

        # Compute drop_connect tensor
        random_tensor = keep_prob
        shape = (tf.shape(inputs)[0],) + (1,) * \
            (len(tf.shape(inputs)) - 1)
        random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output