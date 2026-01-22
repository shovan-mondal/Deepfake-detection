import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time

def run_inference(model_path, image_path):
    # 1. Validation
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 2. Load Model (Only once ideally, but kept here for script structure)
    print("🔄 Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get expected input shape dynamically (Height, Width)
    input_shape = input_details[0]['shape']
    target_height, target_width = input_shape[1], input_shape[2]
    
    print(f"✅ Model loaded. Expected input: {target_height}x{target_width}")

    # 3. Load and Preprocess Image
    print("📷 Loading image...")
    try:
        # convert('RGB') ensures 3 channels even if input is PNG (RGBA) or Grayscale
        img = Image.open(image_path).convert('RGB')
        img = img.resize((target_width, target_height))
        
        # Normalize: Convert to float32 and scale to [-1, 1]
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        
        # Add batch dimension (1, H, W, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return

    # 4. Run Inference
    print("🤖 Running inference...")
    start_time = time.time()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Handle output shape (Flatten in case output is [[0.8]] or [0.8])
    score = output_data.flatten()[0]
    
    inference_time = (time.time() - start_time) * 1000  # ms

    # 5. Interpret Results
    label = "FAKE" if score > 0.5 else "REAL"
    # Calculate confidence (distance from 0.5 decision boundary)
    confidence = score if score > 0.5 else (1 - score)

    print("\n" + "="*35)
    print(f"Prediction:      {label}")
    print(f"Confidence:      {confidence * 100:.2f}%")
    print(f"Raw Output:      {score:.4f}")
    print(f"Inference Time:  {inference_time:.2f} ms")
    print("="*35 + "\n")

# --- Configuration ---
MODEL_PATH = "models/deepfake_detector_mobile_float32.tflite"
IMAGE_PATH = "test_images/test1.jpeg"

# Run
run_inference(MODEL_PATH, IMAGE_PATH)