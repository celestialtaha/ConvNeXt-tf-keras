import tensorflow as tf
import tensorflow.keras.layers as layers
from blocks import *

def create_convnext_model(input_shape=(224, 224, 3), depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=1000, drop_path=0., layer_scale_init_value=1e-6):
    """ Function to construct the ConvNeXt Model
        
        Args:
            input_shape (tuple): (Width, Height , Channels)
            depths (list): a list of size 4. denoting each stage's depth
            dims (list): a list of size 4. denoting number of kernel's in each stage
            num_classes (int): the number of classes
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        Returns:
            ConvNeXt model: an instance of tf.keras.Model
    """

    assert (len(depths) == 4 and len(dims) ==4), "Must provide exactly 4 depths and 4 dims"
    assert (len(input_shape) == 3), "Input shape must be (W, H, C)" 

    input = layers.Input(shape=input_shape)

    # Stem + res2
    y = layers.Conv2D(dims[0], kernel_size=4, strides=4)(input)
    y = layers.LayerNormalization(epsilon=1e-6)(y)
    for i in range(depths[0]):
        y = ConvNeXt_Block(dims[0], drop_path, layer_scale_init_value)(y)

    # downsample + res3
    y = Downsample_Block(dims[1])(y)
    for i in range(depths[1]):
        y = ConvNeXt_Block(dims[1], drop_path, layer_scale_init_value)(y)

    # downsample + res4
    y = Downsample_Block(dims[2])(y)
    for i in range(depths[2]):
        y = ConvNeXt_Block(dims[2], drop_path, layer_scale_init_value)(y)
    
    # downsample + res5
    y = Downsample_Block(dims[3])(y)
    for i in range(depths[3]):
        y = ConvNeXt_Block(dims[3], drop_path, layer_scale_init_value)(y)

    y = layers.GlobalAveragePooling2D()(y)
    # final norm layer
    y = layers.LayerNormalization(epsilon=1e-6)(y) 
    # Head
    y = layers.Dense(num_classes)(y)

    return tf.keras.Model(inputs=input, outputs=y)
