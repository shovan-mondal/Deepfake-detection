import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/best_v3_model_best.keras"
IMAGE_PATH = "test_images/test1.jpeg"

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("📷 Loading image...")
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

print("🤖 Predicting...")
prediction = model.predict(img_array)[0][0]

label = "FAKE" if prediction > 0.5 else "REAL"
confidence = prediction if prediction > 0.5 else (1 - prediction)

print("===================================")
print("Prediction:", label)
print("Confidence:", round(confidence * 100, 2), "%")
print("===================================")
