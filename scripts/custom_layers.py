import tensorflow as tf
from tensorflow.keras import layers

try:
    from tensorflow.keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="waste_classifier")
class RandomHue(layers.Layer):

    def __init__(self, factor=0.05, **kwargs):
        super().__init__(**kwargs)
        # factor of max_delta for tf.image.random_hue, must be in range [0, 0.5]
        self.factor = factor

    def call(self, images, training=None):
        if training:
            return tf.image.random_hue(images, max_delta=self.factor)
        return images

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config