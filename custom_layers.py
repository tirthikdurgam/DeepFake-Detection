from tensorflow.keras.saving import register_keras_serializable # type: ignore
import tensorflow as tf

@register_keras_serializable()
class MyGlobalAveragePoolingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2])  # Example of global average pooling
