import tensorflow as tf
from tensorflow.keras import layers

# register_keras_serializable nằm ở 2 chỗ khác nhau tùy phiên bản:
#   - Keras 3 (TF >= 2.16 mặc định): tf.keras.saving.register_keras_serializable
#   - Keras 2 (TF < 2.16, hoặc TF 2.16+ cài kèm tf-keras): tf.keras.utils.register_keras_serializable
# Thử cả 2 để chạy được trên mọi máy.
try:
    from tensorflow.keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="waste_classifier")
class RandomHue(layers.Layer):
    """
    Random hue augmentation, tự viết bằng tf.image.random_hue vì
    tf.keras.layers.RandomHue chỉ có ở Keras 3.8+, không chắc có sẵn ở
    mọi phiên bản TensorFlow/Keras.
    """

    def __init__(self, factor=0.05, **kwargs):
        super().__init__(**kwargs)
        # factor là max_delta cho tf.image.random_hue, phải nằm trong [0, 0.5]
        self.factor = factor

    def call(self, images, training=None):
        if training:
            return tf.image.random_hue(images, max_delta=self.factor)
        return images

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config