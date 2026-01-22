# Deepfake Detection Backend

A high-performance deepfake detection system using a Dual-Stream Gated Architecture combining MobileNetV2 with forensic SRM filters.

## 🏗️ Project Structure

```
deepfake-backend/
├── app.py                 # FastAPI server for inference
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── Makefile              # Build automation
│
├── src/                  # Source code
│   ├── data/             # Data loading & preprocessing
│   ├── features/         # Feature engineering & augmentation
│   ├── models/           # Model architecture & training
│   │   ├── srm_model.py      # Forensic SRM layers
│   │   ├── train_model.py    # V3 training pipeline
│   │   ├── fine_tune.py      # Fine-tuning pipeline
│   │   └── evaluate_model.py # Evaluation & metrics
│   └── visualization/    # Plotting utilities
│
├── models/               # Trained model files (.keras, .h5, .tflite)
├── tests/                # Test scripts
├── notebooks/            # Jupyter notebooks for exploration
├── reports/              # Generated reports & figures
├── docs/                 # Documentation
├── test_images/          # Sample images for testing
└── config/               # Configuration files
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd deepfake-backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Run API Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

### 3. Test Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@test_images/test1.jpeg"
```

## 🧠 Model Training

### Train V3 Model (Phase 1)

```bash
python -c "from src.models.train_model import main; main()"
```

### Fine-tune (Phase 2)

```bash
python -c "from src.models.fine_tune import fine_tune_v3; fine_tune_v3()"
```

### Evaluate

```bash
python -c "from src.models.evaluate_model import evaluate_model_v3; evaluate_model_v3()"
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Predict if image is real or fake |

## 🔧 Configuration

- Model path: `models/deepfake_detector_mobile_float32.tflite`
- Input size: 224x224 RGB
- Output: `{"prediction": "REAL/FAKE", "confidence": 95.5}`

## 📁 Data

Place your training data in the `data/` directory:
```
data/
├── raw/           # Original datasets
└── processed/     # Preprocessed train/val/test splits
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    └── test/
```
