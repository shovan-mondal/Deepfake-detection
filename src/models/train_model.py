# This is a sample model training script for deepfake detection using TensorFlow and Keras.

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import MobileNetV2
from src.features.build_features import get_preprocessing_pipeline

# Load dataset from processed folders
def load_dataset(data_dir, img_size=(224, 224), batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    return train_ds, val_ds, test_ds


# --- Apply preprocessing and augmentation to train set only ---
def prepare_datasets(train_ds, val_ds, test_ds):
    aug = get_preprocessing_pipeline()
    train_ds = train_ds.map(lambda x, y: (aug(x), y))
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
    test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

    AUTOTUNE = tf.data.AUTOTUNE
    return (
        train_ds.cache().prefetch(AUTOTUNE),
        val_ds.cache().prefetch(AUTOTUNE),
        test_ds.cache().prefetch(AUTOTUNE),
    )


# --- Build Model ---
def build_model(input_shape=(224,224,3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Transfer learning

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- Main Training Script ---
def main():
    data_dir = "data/processed"
    train_ds, val_ds, test_ds = load_dataset(data_dir)
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)

    model = build_model()
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    model.save("models/deepfake_detector_augmented.h5")
    print("✅ Model training complete and saved!")


if __name__ == "__main__":
    main()
