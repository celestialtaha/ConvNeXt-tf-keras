import tensorflow as tf
import tensorflow.keras.layers as layers
from blocks import *

def create_convnext_model(input_shape=(224, 224, 3), depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=1000):

    input = layers.Input(shape=input_shape)

    # Stem + res2
    y = layers.Conv2D(dims[0], kernel_size=4, strides=4)(input)
    y = layers.LayerNormalization(epsilon=1e-6)(y)
    for i in range(depths[0]):
        y = ConvNeXt_Block(dims[0])(y)

    # downsample + res3
    y = Downsample_Block(dims[1])(y)
    for i in range(depths[1]):
        y = ConvNeXt_Block(dims[1])(y)

    # downsample + res4
    y = Downsample_Block(dims[2])(y)
    for i in range(depths[2]):
        y = ConvNeXt_Block(dims[2])(y)
    
    # downsample + res5
    y = Downsample_Block(dims[3])(y)
    for i in range(depths[3]):
        y = ConvNeXt_Block(dims[3])(y)

    y = layers.GlobalAveragePooling2D()(y)
    # final norm layer
    y = layers.LayerNormalization(epsilon=1e-6)(y) 
    # Head
    y = layers.Dense(num_classes)(y)

    return tf.keras.Model(inputs=input, outputs=y)
