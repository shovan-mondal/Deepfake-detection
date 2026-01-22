import tensorflow as tf
from tensorflow.keras import layers

def get_preprocessing_pipeline():
    return tf.keras.Sequential([
        # Geometry is fine (It moves pixels but doesn't change their noise values much)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        
        # Color is fine
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        
        # REMOVED: GaussianNoise. 
        # It confuses the Forensic SRM layer.
        # layers.Lambda(lambda x: x + tf.random.normal(...)) 
        
        # Keep the clip just in case contrast/brightness pushes out of bounds
        layers.Lambda(lambda x: tf.clip_by_value(x, 0, 255), name="clip_values")
        
    ], name="augmentation_pipeline")