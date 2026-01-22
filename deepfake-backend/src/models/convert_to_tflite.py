"""
TFLite Conversion Script for Deepfake Detection Model V3
=========================================================
Converts the Keras model to TensorFlow Lite format for mobile deployment.

Features:
- Supports custom layers (SRMLayerV3, ForensicStreamV3, BayarConv2D, etc.)
- Creates both float32 and quantized (INT8) versions
- Validates converted model accuracy
- Compares inference results between Keras and TFLite

Usage:
    python src/models/convert_to_tflite.py
    
    # Or with custom paths:
    python src/models/convert_to_tflite.py --model models/best_v3_model_best.keras --output models/
"""

import sys
import os
import gc
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import time

import tensorflow as tf

# Memory safety
gc.collect()
tf.keras.backend.clear_session()

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import custom layers
from src.models.srm_model import SRMLayerV3, ForensicStreamV3, BayarConv2D
from src.models.train_model import (
    ChannelAttention, 
    RandomCutout,
    WarmupCosineDecay,
    unnormalize_image
)

# ============================================
# CONFIGURATION
# ============================================

class ConversionConfig:
    """TFLite conversion configuration."""
    
    # Default paths
    DEFAULT_MODEL_PATH = "models/best_v3_model_best.keras"
    OUTPUT_DIR = "models"
    
    # Input specifications
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (1, 224, 224, 3)  # Batch of 1
    
    # Quantization settings
    NUM_CALIBRATION_SAMPLES = 200  # For INT8 quantization
    
    # Output filenames
    FLOAT32_FILENAME = "deepfake_detector_mobile_float32.tflite"
    FLOAT16_FILENAME = "deepfake_detector_mobile_float16.tflite"
    INT8_FILENAME = "deepfake_detector_mobile_int8.tflite"
    DYNAMIC_FILENAME = "deepfake_detector_mobile_dynamic.tflite"


# ============================================
# CUSTOM OBJECTS REGISTRY
# ============================================

def get_custom_objects() -> Dict:
    """Get all custom objects needed for model loading."""
    return {
        'SRMLayerV3': SRMLayerV3,
        'ForensicStreamV3': ForensicStreamV3,
        'BayarConv2D': BayarConv2D,
        'ChannelAttention': ChannelAttention,
        'RandomCutout': RandomCutout,
        'WarmupCosineDecay': WarmupCosineDecay,
        'unnormalize_image': unnormalize_image
    }


# ============================================
# MODEL LOADING
# ============================================

def load_keras_model(model_path: str) -> tf.keras.Model:
    """
    Load the Keras model with custom objects.
    
    Args:
        model_path: Path to the .keras or .h5 model file
        
    Returns:
        Loaded Keras model
    """
    print(f"\n📂 Loading Keras model from: {model_path}")
    
    custom_objects = get_custom_objects()
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("   ✅ Model loaded successfully!")
        
        # Print model summary
        print(f"\n📊 Model Summary:")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        raise


# ============================================
# REPRESENTATIVE DATASET GENERATOR
# ============================================

def create_representative_dataset(
    data_dir: str = "data/processed/test",
    num_samples: int = ConversionConfig.NUM_CALIBRATION_SAMPLES
):
    """
    Create a representative dataset generator for INT8 quantization.
    
    This is required for full integer quantization to calibrate the model.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    print(f"\n📊 Creating representative dataset from: {data_dir}")
    
    # Load some test images
    if os.path.exists(data_dir):
        # Use the test dataset
        test_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=ConversionConfig.IMG_SIZE,
            batch_size=1,
            label_mode=None,
            shuffle=True,
            seed=42
        )
        
        # Take samples
        sample_images = []
        for i, img in enumerate(test_ds.take(num_samples)):
            sample_images.append(preprocess_input(img.numpy()))
        
        print(f"   ✅ Loaded {len(sample_images)} calibration samples")
        
        def representative_dataset_gen():
            for img in sample_images:
                yield [img.astype(np.float32)]
        
        return representative_dataset_gen
    else:
        print(f"   ⚠️ Data directory not found. Using random samples for calibration.")
        
        def representative_dataset_gen():
            for _ in range(num_samples):
                # Generate random images in [-1, 1] range (after preprocess_input)
                data = np.random.uniform(-1, 1, ConversionConfig.INPUT_SHAPE).astype(np.float32)
                yield [data]
        
        return representative_dataset_gen


# ============================================
# TFLITE CONVERSION FUNCTIONS
# ============================================

def convert_to_tflite_float32(model: tf.keras.Model, output_path: str) -> bytes:
    """
    Convert model to TFLite with float32 precision (no quantization).
    
    This is the most accurate but largest model.
    """
    print(f"\n🔄 Converting to TFLite (float32)...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # No optimizations - keep full float32 precision
    converter.target_spec.supported_types = [tf.float32]
    
    try:
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"   ✅ Float32 model saved: {output_path}")
        print(f"   📦 Size: {size_mb:.2f} MB")
        
        return tflite_model
        
    except Exception as e:
        print(f"   ❌ Float32 conversion failed: {e}")
        raise


def convert_to_tflite_float16(model: tf.keras.Model, output_path: str) -> bytes:
    """
    Convert model to TFLite with float16 quantization.
    
    Reduces model size by ~50% with minimal accuracy loss.
    """
    print(f"\n🔄 Converting to TFLite (float16)...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Float16 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    try:
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"   ✅ Float16 model saved: {output_path}")
        print(f"   📦 Size: {size_mb:.2f} MB")
        
        return tflite_model
        
    except Exception as e:
        print(f"   ❌ Float16 conversion failed: {e}")
        raise


def convert_to_tflite_dynamic(model: tf.keras.Model, output_path: str) -> bytes:
    """
    Convert model to TFLite with dynamic range quantization.
    
    Quantizes weights to INT8 but keeps activations as float.
    Good balance of size and accuracy.
    """
    print(f"\n🔄 Converting to TFLite (dynamic range quantization)...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    try:
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"   ✅ Dynamic range model saved: {output_path}")
        print(f"   📦 Size: {size_mb:.2f} MB")
        
        return tflite_model
        
    except Exception as e:
        print(f"   ❌ Dynamic range conversion failed: {e}")
        raise


def convert_to_tflite_int8(
    model: tf.keras.Model, 
    output_path: str,
    representative_dataset_gen
) -> Optional[bytes]:
    """
    Convert model to TFLite with full INT8 quantization.
    
    Smallest model size, may have accuracy degradation.
    Requires representative dataset for calibration.
    """
    print(f"\n🔄 Converting to TFLite (INT8 full quantization)...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Ensure full integer quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS  # Fallback for unsupported ops
    ]
    
    # Set input/output types
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32  # Keep output as float for probability
    
    try:
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"   ✅ INT8 model saved: {output_path}")
        print(f"   📦 Size: {size_mb:.2f} MB")
        
        return tflite_model
        
    except Exception as e:
        print(f"   ⚠️ INT8 conversion failed: {e}")
        print("   💡 This can happen with complex custom layers.")
        print("   💡 The float32/float16/dynamic models are still usable.")
        return None


# ============================================
# VALIDATION AND TESTING
# ============================================

def run_tflite_inference(
    tflite_model_path: str, 
    input_data: np.ndarray
) -> np.ndarray:
    """
    Run inference on a TFLite model.
    
    Args:
        tflite_model_path: Path to the .tflite file
        input_data: Input image array (1, 224, 224, 3)
        
    Returns:
        Model prediction
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if input needs to be quantized
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.uint8:
        # Quantize input from [-1, 1] to [0, 255]
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
    else:
        input_data = input_data.astype(np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output


def validate_tflite_model(
    keras_model: tf.keras.Model,
    tflite_model_path: str,
    model_name: str,
    num_test_samples: int = 10
) -> Dict:
    """
    Validate TFLite model against original Keras model.
    
    Compares predictions and measures inference speed.
    """
    print(f"\n🧪 Validating {model_name}...")
    
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    # Generate test samples
    np.random.seed(42)
    test_images = []
    for _ in range(num_test_samples):
        # Random images in [0, 255]
        img = np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32)
        # Preprocess to [-1, 1]
        img = preprocess_input(img)
        test_images.append(img)
    
    # Keras predictions
    keras_preds = []
    keras_times = []
    for img in test_images:
        start = time.time()
        pred = keras_model.predict(img, verbose=0)
        keras_times.append(time.time() - start)
        keras_preds.append(pred[0][0])
    
    # TFLite predictions
    tflite_preds = []
    tflite_times = []
    
    # Load interpreter once
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    for img in test_images:
        # Handle quantized input
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.uint8:
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            input_data = ((img / input_scale) + input_zero_point).astype(np.uint8)
        else:
            input_data = img.astype(np.float32)
        
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        tflite_times.append(time.time() - start)
        tflite_preds.append(output.flatten()[0])
    
    # Calculate metrics
    keras_preds = np.array(keras_preds)
    tflite_preds = np.array(tflite_preds)
    
    # Mean absolute error
    mae = np.mean(np.abs(keras_preds - tflite_preds))
    
    # Maximum absolute error
    max_error = np.max(np.abs(keras_preds - tflite_preds))
    
    # Classification agreement (using 0.5 threshold)
    keras_class = (keras_preds >= 0.5).astype(int)
    tflite_class = (tflite_preds >= 0.5).astype(int)
    agreement = np.mean(keras_class == tflite_class) * 100
    
    # Timing
    avg_keras_time = np.mean(keras_times) * 1000  # ms
    avg_tflite_time = np.mean(tflite_times) * 1000  # ms
    speedup = avg_keras_time / avg_tflite_time if avg_tflite_time > 0 else 0
    
    results = {
        'mae': mae,
        'max_error': max_error,
        'agreement': agreement,
        'avg_keras_time_ms': avg_keras_time,
        'avg_tflite_time_ms': avg_tflite_time,
        'speedup': speedup
    }
    
    print(f"   📊 Mean Absolute Error: {mae:.6f}")
    print(f"   📊 Max Absolute Error: {max_error:.6f}")
    print(f"   📊 Classification Agreement: {agreement:.1f}%")
    print(f"   ⏱️ Keras Avg Time: {avg_keras_time:.2f} ms")
    print(f"   ⏱️ TFLite Avg Time: {avg_tflite_time:.2f} ms")
    print(f"   🚀 Speedup: {speedup:.2f}x")
    
    return results


def get_tflite_model_info(tflite_model_path: str) -> Dict:
    """Get detailed information about a TFLite model."""
    
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get file size
    file_size_mb = os.path.getsize(tflite_model_path) / (1024 * 1024)
    
    info = {
        'file_size_mb': file_size_mb,
        'input_shape': input_details[0]['shape'].tolist(),
        'input_dtype': str(input_details[0]['dtype']),
        'output_shape': output_details[0]['shape'].tolist(),
        'output_dtype': str(output_details[0]['dtype']),
    }
    
    # Check if quantized
    if 'quantization' in input_details[0]:
        quant = input_details[0]['quantization']
        if quant[0] != 0:
            info['input_quantization'] = {
                'scale': quant[0],
                'zero_point': quant[1]
            }
    
    return info


# ============================================
# MAIN CONVERSION PIPELINE
# ============================================

def convert_model_to_tflite(
    model_path: str = ConversionConfig.DEFAULT_MODEL_PATH,
    output_dir: str = ConversionConfig.OUTPUT_DIR,
    data_dir: str = "data/processed/test",
    create_all_variants: bool = True,
    validate: bool = True
) -> Dict:
    """
    Main conversion pipeline.
    
    Args:
        model_path: Path to the Keras model
        output_dir: Directory to save TFLite models
        data_dir: Path to test data for calibration
        create_all_variants: Create all quantization variants
        validate: Validate converted models
        
    Returns:
        Dictionary with paths and metrics for each variant
    """
    print("=" * 60)
    print("🔄 TFLITE CONVERSION PIPELINE")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Keras model
    keras_model = load_keras_model(model_path)
    
    # Get original model size
    keras_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n📦 Original Keras model size: {keras_size:.2f} MB")
    
    results = {}
    
    # ============================================
    # Convert to different variants
    # ============================================
    
    # 1. Float32 (always create this)
    float32_path = os.path.join(output_dir, ConversionConfig.FLOAT32_FILENAME)
    try:
        convert_to_tflite_float32(keras_model, float32_path)
        results['float32'] = {'path': float32_path, 'success': True}
    except Exception as e:
        results['float32'] = {'path': None, 'success': False, 'error': str(e)}
    
    if create_all_variants:
        # 2. Float16
        float16_path = os.path.join(output_dir, ConversionConfig.FLOAT16_FILENAME)
        try:
            convert_to_tflite_float16(keras_model, float16_path)
            results['float16'] = {'path': float16_path, 'success': True}
        except Exception as e:
            results['float16'] = {'path': None, 'success': False, 'error': str(e)}
        
        # 3. Dynamic Range
        dynamic_path = os.path.join(output_dir, ConversionConfig.DYNAMIC_FILENAME)
        try:
            convert_to_tflite_dynamic(keras_model, dynamic_path)
            results['dynamic'] = {'path': dynamic_path, 'success': True}
        except Exception as e:
            results['dynamic'] = {'path': None, 'success': False, 'error': str(e)}
        
        # 4. INT8 (Full quantization)
        int8_path = os.path.join(output_dir, ConversionConfig.INT8_FILENAME)
        try:
            rep_dataset = create_representative_dataset(data_dir)
            tflite_int8 = convert_to_tflite_int8(keras_model, int8_path, rep_dataset)
            if tflite_int8:
                results['int8'] = {'path': int8_path, 'success': True}
            else:
                results['int8'] = {'path': None, 'success': False, 'error': 'Conversion returned None'}
        except Exception as e:
            results['int8'] = {'path': None, 'success': False, 'error': str(e)}
    
    # ============================================
    # Validation
    # ============================================
    
    if validate:
        print("\n" + "=" * 60)
        print("🧪 VALIDATION")
        print("=" * 60)
        
        for variant, info in results.items():
            if info['success'] and info['path']:
                try:
                    # Get model info
                    model_info = get_tflite_model_info(info['path'])
                    info['model_info'] = model_info
                    
                    # Validate
                    validation = validate_tflite_model(
                        keras_model, 
                        info['path'], 
                        variant.upper()
                    )
                    info['validation'] = validation
                    
                except Exception as e:
                    print(f"   ⚠️ Validation failed for {variant}: {e}")
                    info['validation'] = {'error': str(e)}
    
    # ============================================
    # Summary
    # ============================================
    
    print("\n" + "=" * 60)
    print("📊 CONVERSION SUMMARY")
    print("=" * 60)
    
    print(f"\n📁 Original Keras Model: {keras_size:.2f} MB")
    print("-" * 40)
    
    for variant, info in results.items():
        if info['success'] and info['path']:
            size = info.get('model_info', {}).get('file_size_mb', 0)
            reduction = ((keras_size - size) / keras_size) * 100 if keras_size > 0 else 0
            agreement = info.get('validation', {}).get('agreement', 'N/A')
            
            status = "✅"
            print(f"{status} {variant.upper():12} | Size: {size:6.2f} MB | "
                  f"Reduction: {reduction:5.1f}% | Agreement: {agreement}%")
        else:
            error = info.get('error', 'Unknown error')[:40]
            print(f"❌ {variant.upper():12} | Failed: {error}...")
    
    print("\n" + "=" * 60)
    print("✅ CONVERSION COMPLETE")
    print("=" * 60)
    
    # Recommendation
    print("\n💡 RECOMMENDATION:")
    if results.get('float16', {}).get('success'):
        print("   For mobile deployment, use float16 model:")
        print(f"   → {results['float16']['path']}")
        print("   Best balance of size, speed, and accuracy.")
    elif results.get('dynamic', {}).get('success'):
        print("   For mobile deployment, use dynamic range model:")
        print(f"   → {results['dynamic']['path']}")
    else:
        print("   For mobile deployment, use float32 model:")
        print(f"   → {results['float32']['path']}")
    
    return results


# ============================================
# UTILITY FUNCTIONS
# ============================================

def test_single_image(
    tflite_model_path: str,
    image_path: str
) -> float:
    """
    Test TFLite model on a single image.
    
    Args:
        tflite_model_path: Path to .tflite model
        image_path: Path to image file
        
    Returns:
        Prediction probability (fake score)
    """
    from PIL import Image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(ConversionConfig.IMG_SIZE)
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Run inference
    prediction = run_tflite_inference(tflite_model_path, img_array)
    
    prob = prediction.flatten()[0]
    label = "FAKE" if prob > 0.5 else "REAL"
    
    print(f"\n🖼️ Image: {image_path}")
    print(f"📊 Prediction: {label} ({prob:.4f})")
    
    return prob


def benchmark_tflite_model(
    tflite_model_path: str,
    num_iterations: int = 100
) -> Dict:
    """
    Benchmark TFLite model inference speed.
    
    Args:
        tflite_model_path: Path to .tflite model
        num_iterations: Number of inference iterations
        
    Returns:
        Benchmark results
    """
    print(f"\n⏱️ Benchmarking: {tflite_model_path}")
    print(f"   Iterations: {num_iterations}")
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        input_data = np.random.randint(0, 256, input_shape).astype(np.uint8)
    else:
        input_data = np.random.uniform(-1, 1, input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    results = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000 / np.mean(times)
    }
    
    print(f"   Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
    print(f"   Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
    print(f"   FPS: {results['fps']:.1f}")
    
    return results


# ============================================
# CLI INTERFACE
# ============================================

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert Deepfake Detection Model to TFLite"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=ConversionConfig.DEFAULT_MODEL_PATH,
        help="Path to Keras model file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=ConversionConfig.OUTPUT_DIR,
        help="Output directory for TFLite models"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="data/processed/test",
        help="Path to test data for calibration"
    )
    parser.add_argument(
        "--float32-only",
        action="store_true",
        help="Only create float32 model (skip quantization)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark a specific TFLite model"
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Test model on a specific image"
    )
    
    args = parser.parse_args()
    
    # Handle benchmark mode
    if args.benchmark:
        benchmark_tflite_model(args.benchmark)
        return
    
    # Handle test image mode
    if args.test_image:
        # Find the best available model
        for variant in ['float16', 'dynamic', 'float32']:
            model_path = os.path.join(args.output, f"deepfake_detector_mobile_{variant}.tflite")
            if os.path.exists(model_path):
                test_single_image(model_path, args.test_image)
                return
        print("❌ No TFLite model found. Run conversion first.")
        return
    
    # Main conversion
    convert_model_to_tflite(
        model_path=args.model,
        output_dir=args.output,
        data_dir=args.data,
        create_all_variants=not args.float32_only,
        validate=not args.no_validate
    )


if __name__ == "__main__":
    main()
