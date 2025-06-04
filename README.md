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
git clone https://github.com/Abdulla-1234/Speech-Fluency-Analysis-Speech-Based-Emotion-Enhanced-Learning.git
cd Speech-Fluency-Analysis-Speech-Based-Emotion-Enhanced-Learning
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
