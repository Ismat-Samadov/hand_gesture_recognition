# Hand Gesture Recognition - End-to-End ML Workflow

A production-ready deep learning system for recognizing hand gestures from near-infrared images captured by Leap Motion sensor.

## 📋 Project Overview

This project implements a comprehensive machine learning workflow following industry best practices for hand gesture recognition. The system can classify 10 different hand gestures with high accuracy using a custom Convolutional Neural Network.

### Gestures Recognized
1. Palm
2. L
3. Fist
4. Fist Moved
5. Thumb
6. Index
7. OK
8. Palm Moved
9. C
10. Down

## 🏗️ Project Structure

```
hand_gesture_recognition/
├── hand_gesture_recognition.ipynb   # Main notebook with complete workflow
├── requirements.txt                 # Python dependencies
├── dataset/                         # Hand gesture image dataset
│   ├── 00/                         # Subject 00
│   ├── 01/                         # Subject 01
│   └── ...
├── charts/                         # Generated visualizations
│   ├── data_distribution.png
│   ├── sample_gestures.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── per_class_accuracy.png
│   └── sample_predictions.png
├── outputs/                        # Evaluation results and metrics
│   ├── validation_report.json
│   ├── image_statistics.csv
│   ├── split_distribution.csv
│   ├── model_architecture.txt
│   ├── training_log.csv
│   ├── classification_report.txt
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── per_class_metrics.csv
│   └── evaluation_summary.json
└── artifacts/                      # Model and preprocessing artifacts
    ├── config.json
    ├── label_encoder.pkl
    ├── best_model.keras
    ├── final_model.keras
    ├── model_weights.h5
    ├── saved_model/                # TensorFlow SavedModel format
    ├── training_history.json
    └── preprocessing_params.json
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- GPU optional (but recommended for faster training)

### Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Start Jupyter Notebook:

```bash
jupyter notebook
```

2. Open `hand_gesture_recognition.ipynb`
3. Run all cells sequentially (Cell > Run All)

The notebook will:
- Create necessary directories
- Load and validate the dataset
- Perform exploratory data analysis
- Preprocess images
- Split data into train/validation/test sets
- Build and train a CNN model
- Evaluate model performance
- Generate visualizations and reports
- Save all artifacts for production use

## 📊 Model Performance

The trained model achieves:
- **High accuracy** on multi-class gesture classification
- **Balanced performance** across all gesture classes
- **Robust predictions** with confidence scores

Detailed metrics are saved in `outputs/evaluation_summary.json`

## 🔑 Key Features

### Reproducibility
- Fixed random seeds across all libraries
- Configuration-driven parameters
- Environment-agnostic paths
- Complete dependency specification

### Data Handling
- Stratified train/validation/test splits (70/15/15)
- No data leakage between sets
- Comprehensive data validation
- Image preprocessing pipeline

### Model Development
- Custom CNN architecture optimized for grayscale images
- Batch normalization and dropout for regularization
- Data augmentation for training robustness
- Learning rate scheduling and early stopping

### Evaluation
- Multiple metrics: accuracy, precision, recall, F1-score
- Per-class performance analysis
- Confusion matrices (raw and normalized)
- Sample prediction visualizations

### Production Ready
- Multiple model save formats (Keras, SavedModel, weights)
- Serialized preprocessing parameters
- Label encoder for inference
- Complete inference pipeline example

## 📈 Visualizations

All visualizations are automatically generated and saved to the `charts/` directory:

1. **Data Distribution**: Class and subject distribution
2. **Sample Gestures**: Example images from each class
3. **Training History**: Accuracy, loss, precision, recall over epochs
4. **Confusion Matrix**: Model predictions vs. true labels
5. **Per-Class Accuracy**: Individual class performance
6. **Sample Predictions**: Visual verification of model predictions

## 🔧 Configuration

Key parameters can be modified in the `Config` class within the notebook:

- `RANDOM_SEED`: For reproducibility (default: 42)
- `IMG_HEIGHT`, `IMG_WIDTH`: Image dimensions (default: 128x128)
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`: Data split ratios
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Maximum training epochs (default: 50)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)

## 📦 Model Deployment

The trained model can be deployed using:

1. **Keras Model**: Load with `keras.models.load_model('artifacts/final_model.keras')`
2. **TensorFlow Serving**: Use `artifacts/saved_model/` directory
3. **Custom API**: Use the provided `predict_gesture()` function

Example inference:

```python
from tensorflow import keras
import joblib

# Load model and encoder
model = keras.models.load_model('artifacts/final_model.keras')
encoder = joblib.load('artifacts/label_encoder.pkl')

# Preprocess and predict
# (see notebook for complete inference pipeline)
```

## 📚 Dataset Reference

**Citation**: T. Mantecón, C.R. del Blanco, F. Jaureguizar, N. García, "Hand Gesture Recognition using Infrared Imagery Provided by Leap Motion Controller", Int. Conf. on Advanced Concepts for Intelligent Vision Systems, ACIVS 2016, Lecce, Italy, pp. 47-57, 24-27 Oct. 2016. (doi: 10.1007/978-3-319-48680-2_5)

## 🤝 Contributing

This is a complete, self-contained project designed for educational and production use. Feel free to:
- Experiment with different model architectures
- Tune hyperparameters
- Add additional evaluation metrics
- Implement real-time gesture recognition

## 📝 License

This project is provided as-is for educational and research purposes.

## ✅ Checklist for Production Deployment

- [x] Data validation and quality checks
- [x] Reproducible preprocessing pipeline
- [x] Stratified data splitting
- [x] Model training with monitoring
- [x] Comprehensive evaluation
- [x] Model serialization
- [x] Documentation
- [ ] API endpoint implementation
- [ ] Model monitoring setup
- [ ] A/B testing framework
- [ ] Production data pipeline
- [ ] Model retraining schedule

---

**Note**: This notebook is designed to be fully self-contained and reproducible. All random operations are seeded, all paths are relative, and all dependencies are specified.
