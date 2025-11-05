import tensorflow as tf
from tensorflow.keras import layers

def get_preprocessing_pipeline():
    """
    Returns a Sequential Keras pipeline that performs
    data normalization and augmentation.
    """
    return tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomShear(0.1),
        layers.GaussianNoise(0.05),
    ], name="augmentation_pipeline")
