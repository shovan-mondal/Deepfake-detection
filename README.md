# 🕵️‍♂️ Deepfake Image Detection V3

A state-of-the-art (SOTA) deepfake detection system designed to distinguish between real and AI-generated images with **99.13% AUC**.

This project utilizes a **Dual-Stream Gated Architecture** that simultaneously analyzes high-level visual features (faces, expressions) and low-level forensic artifacts (GAN fingerprints, noise patterns), making it robust against high-quality Deepfakes (Midjourney, Stable Diffusion, DeepFaceLab).

## Key Features

* **Dual-Stream Architecture:** Combines a **MobileNetV2** backbone (Visual Stream) with custom **SRM & Bayar Conv layers** (Forensic Stream).
* **Adaptive Gating:** A learned gating mechanism dynamically weights the importance of visual vs. forensic features for each image.
* **Mobile Optimized:** Ultra-lightweight design (~4.02M parameters) compatible with **TensorFlow Lite (TFLite)** for edge deployment (~16MB file size).
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
deepfake_image_detection
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources (CIFAKE, FF++, Celeb-DF).
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models (keras/tflite)
│   ├── best_v3_model.keras  <- The production SOTA model
│   └── model_v3.tflite      <- Mobile-optimized version
│
├── notebooks          <- Jupyter notebooks for exploration and prototyping.
│
├── references         <- Data dictionaries, manuals, and explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics (ROC curves, Confusion Matrices).
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
├── setup.py           <- Makes project pip installable (pip install -e .)
├── src                <- Source code for use in this project.
│   ├── __init__.py
│   ├── data           <- Scripts to download or generate data
│   ├── features       <- Scripts to turn raw data into features for modeling
│   ├── models         <- Scripts to train models and make predictions
│   │   ├── train_model.py  <- Main training script (Dual-Stream V3)
│   │   ├── predict_model.py <- Inference script for batches
│   │   └── srm_model.py     <- Custom layers (SRM/Bayar) definition
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│
└── tox.ini            <- tox file with settings for running tox

```

## Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/shovan-mondal/Deepfake-detection.git
cd deepfake-image-detection

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run Training (V3):**
```bash
python src/models/train_model.py

```


*This will automatically load data, build the Dual-Stream V3 model, and save the best checkpoint.*
4. **Run "Wild" Inference (Test on single image):**
```python
# See 'notebooks' or run the inference script
python src/models/predict_model.py --image path/to/image.jpg

```



## Mobile Deployment

The model is fully compatible with TensorFlow Lite.

* **Float32 Size:** 44.68 MB
* **Quantized Size:** 4.36 MB

##  License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/shovan-mondal/Deepfake-detection/blob/main/LICENSE) file for details.

---

Project based on the cookiecutter data science project template [#cookiecutterdatascience](https://drivendata.github.io/cookiecutter-data-science/)