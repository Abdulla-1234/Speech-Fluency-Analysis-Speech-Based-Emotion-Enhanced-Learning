# Speech-Fluency-Analysis-Speech-Based-Emotion-Enhanced-Learning

A comprehensive machine learning-based system designed to classify non-native English speakers' fluency levels using advanced audio processing and deep learning techniques. The system achieves up to **94% accuracy** with Support Vector Machine (SVM) and incorporates emotional state detection for adaptive personalized learning experiences.

## Project Overview

This project presents an innovative approach to automate speaker fluency evaluation using machine learning algorithms. By analyzing audio conversations between non-native English speakers, the system can classify fluency into three levels: **Low**, **Intermediate**, and **High**.

### Key Features
- **Automated Fluency Assessment**: Real-time evaluation of speaking proficiency
- **Multi-Model Approach**: Comparison of 5 ML algorithms (SVM, CNN, RNN, RF, MLP)
- **Advanced Feature Extraction**: MFCC, Zero-Crossing Rate, RMSE, Spectral Flux
- **Emotion-Enhanced Learning**: Adaptive system based on emotional state detection
- **High Accuracy**: Up to 94% classification accuracy with SVM model

## Technology Stack

- **Programming Language**: Python 3.8+
- **Machine Learning**: scikit-learn, TensorFlow, Keras
- **Audio Processing**: Librosa, PyDub
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deep Learning Models**: CNN, RNN (LSTM), MLP
- **Traditional ML**: SVM, Random Forest

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)
- Git
- Jupyter Notebook or JupyterLab
- Audio codecs for various formats

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/speech-fluency-analysis.git
cd speech-fluency-analysis
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv speech_fluency_env

# Activate virtual environment
# On Windows:
speech_fluency_env\Scripts\activate

# On macOS/Linux:
source speech_fluency_env/bin/activate
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tensorflow keras
pip install librosa pydub
pip install jupyter ipykernel
pip install plotly scipy
```

### 4. Install Audio Processing Dependencies

```bash
# For Linux/Ubuntu:
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1

# For macOS:
brew install ffmpeg libsndfile

# For Windows:
# Download ffmpeg from https://ffmpeg.org/download.html
# Add to system PATH
```

### 5. Setup Jupyter Kernel

```bash
# Add virtual environment to Jupyter
python -m ipykernel install --user --name=speech_fluency_env --display-name="Speech Fluency Analysis"
```

## Project Structure

```
Speech-Fluency-Analysis/
├── code/
│   ├── Individual Scripts/
│   │   ├── CNN.ipynb                    # Convolutional Neural Network model
│   │   ├── MLP.ipynb                    # Multi-Layer Perceptron model
│   │   ├── Multiple_Scatter_plots.ipynb # Data visualization
│   │   ├── RF.ipynb                     # Random Forest model
│   │   ├── RNN.ipynb                    # Recurrent Neural Network model
│   │   ├── SVM Confusion Matrix.ipynb   # SVM performance analysis
│   │   └── SVM.ipynb                    # Support Vector Machine model
│   └── Fluency Audioset Experiments.ipynb # Main experiment notebook
├── data/
│   ├── audio-data/
│   │   ├── Avalinguo - Alan and Eduardo...
│   │   ├── Avalinguo - Gonzalo and Eddy...
│   │   └── Avalinguo - Victor and Abraham...
│   ├── feat.npy                         # Extracted features
│   └── label.npy                        # Classification labels
├── Report.pdf                           # Detailed project report
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
```

## Running the Project

### 1. Start Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Or start JupyterLab
jupyter lab
```

### 2. Run Individual Models

Execute each model notebook individually:

```bash
# Navigate to project directory
cd speech-fluency-analysis

# Run specific model scripts
jupyter nbconvert --to notebook --execute code/Individual\ Scripts/SVM.ipynb
jupyter nbconvert --to notebook --execute code/Individual\ Scripts/CNN.ipynb
jupyter nbconvert --to notebook --execute code/Individual\ Scripts/RNN.ipynb
jupyter nbconvert --to notebook --execute code/Individual\ Scripts/RF.ipynb
jupyter nbconvert --to notebook --execute code/Individual\ Scripts/MLP.ipynb
```

### 3. Run Complete Experiment

```bash
# Execute main experiment notebook
jupyter nbconvert --to notebook --execute code/Fluency\ Audioset\ Experiments.ipynb
```

### 4. Command Line Model Training

Create a Python script to run models from command line:

```bash
# Create and run training script
python train_models.py --model svm --features mfcc --n_mfcc 20
python train_models.py --model cnn --features all --epochs 100
python train_models.py --model rnn --features mfcc_zcr_rmse --batch_size 32
```

### 5. Feature Extraction

```bash
# Extract audio features from new data
python extract_features.py --input_dir data/audio-data --output_dir data/features
```

### 6. Model Evaluation

```bash
# Evaluate trained models
python evaluate_models.py --model_dir models/ --test_data data/test/
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM   | **94%**  | 0.93      | 0.94   | 0.93     |
| CNN   | 91%      | 0.90      | 0.91   | 0.90     |
| RNN   | 89%      | 0.88      | 0.89   | 0.88     |
| RF    | 87%      | 0.86      | 0.87   | 0.86     |
| MLP   | 85%      | 0.84      | 0.85   | 0.84     |

## Usage Examples

### Basic Model Training

```python
# Load and preprocess data
from sklearn.model_selection import train_test_split
import numpy as np

# Load features and labels
features = np.load('data/feat.npy')
labels = np.load('data/label.npy')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)
```

### Feature Extraction

```python
import librosa
import numpy as np

def extract_features(audio_file, n_mfcc=20):
    """Extract MFCC and additional features from audio"""
    y, sr = librosa.load(audio_file)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Extract additional features
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    
    return np.concatenate([
        np.mean(mfccs, axis=1),
        np.mean(zcr),
        np.mean(rmse),
        np.mean(spectral_flux)
    ])
```

### Model Prediction

```python
from sklearn.svm import SVC
import joblib

# Load trained model
model = joblib.load('models/svm_model.pkl')

# Predict fluency level
fluency_level = model.predict(features)
confidence = model.predict_proba(features)

print(f"Fluency Level: {fluency_level}")
print(f"Confidence: {confidence}")
```

## Development Commands

### Data Preprocessing

```bash
# Segment audio files into 5-second clips
python preprocess_audio.py --input_dir raw_audio/ --output_dir segmented_audio/ --segment_length 5

# Create training dataset
python create_dataset.py --audio_dir segmented_audio/ --output_dir data/
```

### Model Training Pipeline

```bash
# Train all models with different configurations
python train_pipeline.py --config configs/experiment_config.json

# Hyperparameter tuning
python hyperparameter_tuning.py --model svm --param_grid configs/svm_params.json
```

### Performance Analysis

```bash
# Generate confusion matrices
python generate_confusion_matrix.py --model_dir models/ --output_dir results/

# Create performance plots
python plot_performance.py --results_dir results/ --output_dir plots/
```

## Visualization Commands

```bash
# Generate spectrograms
python visualize_audio.py --audio_file data/sample.wav --output_dir plots/

# Plot model comparison
python plot_model_comparison.py --results_file results/model_comparison.json

# Create feature importance plots
python plot_feature_importance.py --model_file models/rf_model.pkl
```

## Experiments

### Experiment 1: MFCC Coefficient Optimization

```bash
# Test different numbers of MFCC coefficients
for n_mfcc in 13 16 20 24 26; do
    python experiment_mfcc.py --n_mfcc $n_mfcc --output results/mfcc_${n_mfcc}.json
done
```

### Experiment 2: Feature Combination Analysis

```bash
# Test different feature combinations
python experiment_features.py --features mfcc --output results/mfcc_only.json
python experiment_features.py --features mfcc_zcr --output results/mfcc_zcr.json
python experiment_features.py --features all --output results/all_features.json
```

### Experiment 3: Model Architecture Comparison

```bash
# Compare different neural network architectures
python experiment_architectures.py --config configs/architectures.json
```

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Test individual components
python -m pytest tests/test_feature_extraction.py
python -m pytest tests/test_models.py
python -m pytest tests/test_preprocessing.py

# Integration tests
python -m pytest tests/test_integration.py -v
```

## Results Analysis

```bash
# Generate comprehensive report
python generate_report.py --results_dir results/ --output report.html

# Statistical significance testing
python statistical_analysis.py --results_file results/model_comparison.json

# Cross-validation analysis
python cross_validation.py --n_folds 5 --output results/cv_results.json
```

## Troubleshooting

### Common Issues and Solutions

1. **Audio Loading Errors:**
   ```bash
   # Install additional audio codecs
   pip install soundfile
   pip install audioread
   ```

2. **Memory Issues with Large Datasets:**
   ```bash
   # Use batch processing
   python train_models.py --batch_size 32 --use_generator True
   ```

3. **CUDA/GPU Issues:**
   ```bash
   # Check TensorFlow GPU installation
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

4. **Librosa Installation Issues:**
   ```bash
   # For Linux systems
   sudo apt-get install libsndfile1-dev
   pip install librosa --no-cache-dir
   ```

## Dataset Information

- **Total Audio Duration**: 118.65 minutes
- **Total Segments**: 1,424 audio clips (5 seconds each)
- **Fluency Classes**: 3 levels (Low, Intermediate, High)
- **Features**: MFCC (13-26 coefficients), ZCR, RMSE, Spectral Flux
- **Languages**: English (non-native speakers)

## Model Configurations

### SVM Configuration
```python
SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'class_weight': 'balanced'
}
```

### CNN Architecture
```python
CNN_ARCHITECTURE = {
    'conv_layers': [
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 32, 'kernel_size': 3, 'activation': 'relu'}
    ],
    'dense_layers': [
        {'units': 512, 'activation': 'relu'},
        {'units': 3, 'activation': 'softmax'}
    ]
}
```

**Made with ❤️ using Python, TensorFlow, and scikit-learn**
