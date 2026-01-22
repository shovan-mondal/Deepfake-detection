# 🕵️‍♂️ Deepfake Image Detection V3

A state-of-the-art (SOTA) deepfake detection system designed to distinguish between real and AI-generated images with **99.13% AUC**.

This project utilizes a **Dual-Stream Gated Architecture** that simultaneously analyzes high-level visual features (faces, expressions) and low-level forensic artifacts (GAN fingerprints, noise patterns), making it robust against high-quality Deepfakes (Midjourney, Stable Diffusion, DeepFaceLab).

## Key Features

* **Dual-Stream Architecture:** Combines a **MobileNetV2** backbone (Visual Stream) with custom **SRM & Bayar Conv layers** (Forensic Stream).
* **Adaptive Gating:** A learned gating mechanism dynamically weights the importance of visual vs. forensic features for each image.
* **Mobile Optimized:** Ultra-lightweight design (~4.02M parameters) compatible with **TensorFlow Lite (TFLite)** for edge deployment (4.36 MB file size).
* **"Wild" Generalization:** Proven capability to detect unseen, real-world viral deepfakes (e.g., Rashmika Mandanna, Jake Gyllenhaal) with >96% confidence.
* **Robust Training:** Trained on a diverse mix of **CIFAKE**, **FaceForensics++**, and **Celeb-DF** datasets to prevent overfitting to specific artifacts.

## Performance Metrics (V3)

Evaluated on a held-out test set of 23,564 images.

| Metric | Score | Description |
| --- | --- | --- |
| **AUC-ROC** | **0.9913** | Exceptional separation between Real and Fake classes. |
| **Accuracy** | **94.78%** | Overall correctness on the test dataset. |
| **Precision** | **93.46%** | Low false positive rate (rarely calls a real image "Fake"). |
| **Recall** | **96.21%** | High sensitivity (catches 96% of deepfakes). |
| **F1-Score** | **0.9481** | Balanced performance between precision and recall. |

## Model Architecture

The model uses a **V3 Dual-Stream approach**:

1. **Visual Stream (RGB):** Uses `MobileNetV2` to extract semantic features (eyes, mouth, facial structure).
2. **Forensic Stream (Noise):** Uses a custom `SRMLayer` (Spatial Rich Model) and `BayarConv2D` to suppress image content and isolate noise residuals (pixel-level artifacts).
3. **Fusion & Gating:** The two streams are concatenated and weighted by a gating network before the final classification head.

## Project Organization

```text
deepfake-image-detection/
├── deepfake-frontend/     <- React + Vite frontend application
│   ├── src/               <- React components
│   ├── package.json       <- Node.js dependencies
│   └── README.md
│
├── deepfake-backend/      <- Python ML backend
│   ├── app.py             <- FastAPI inference server
│   ├── requirements.txt   <- Python dependencies
│   ├── src/               <- Source code for ML pipeline
│   │   ├── data/          <- Data loading & preprocessing
│   │   ├── features/      <- Feature engineering & augmentation
│   │   ├── models/        <- Model architecture & training
│   │   │   ├── srm_model.py      <- SRM & Bayar forensic layers
│   │   │   ├── train_model.py    <- V3 training pipeline
│   │   │   ├── fine_tune.py      <- Fine-tuning pipeline
│   │   │   └── evaluate_model.py <- Evaluation & metrics
│   │   └── visualization/ <- Plotting utilities
│   ├── models/            <- Trained models (.keras, .tflite)
│   ├── tests/             <- Test scripts
│   ├── notebooks/         <- Jupyter notebooks
│   ├── reports/           <- Generated reports & figures
│   └── docs/              <- Documentation
│
├── data/                  <- Training datasets (external, not in repo)
│   ├── raw/               <- Original datasets
│   └── processed/         <- Train/val/test splits
│
├── LICENSE
└── README.md

## Installation & Usage

### Backend Setup

```bash
cd deepfake-backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Start API Server

```bash
cd deepfake-backend
uvicorn app:app --reload --port 8000
```

### Frontend Setup (requires Node.js 20+)

```bash
cd deepfake-frontend
npm install
npm run dev
```

### Run Training (V3)

```bash
cd deepfake-backend
python -c "from src.models.train_model import main; main()"
```

### Run Inference (Single Image)

```bash
cd deepfake-backend
python tests/test_predict.py
```

Or via API:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@test_images/test1.jpeg"
```



## Mobile Deployment

The model is fully compatible with TensorFlow Lite.

* **Float32 Size:** 15.53 MB
* **Quantized Size:** 4.36 MB

##  License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/shovan-mondal/Deepfake-detection/blob/main/LICENSE) file for details.

---

Project based on the cookiecutter data science project template [#cookiecutterdatascience](https://drivendata.github.io/cookiecutter-data-science/)