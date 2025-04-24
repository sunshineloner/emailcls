# Email Spam Classifier

A deep learning-based email spam classifier using PyTorch. This project implements a neural network model that classifies emails as spam or ham (non-spam) using TF-IDF features and a three-layer neural network.

## Features

- Text preprocessing using NLTK
- TF-IDF feature extraction with n-gram support
- Neural network model with batch normalization and dropout
- Training pipeline with learning rate scheduling
- Comprehensive evaluation metrics and visualizations
- Support for both CPU and GPU (CUDA)
- Model and vectorizer persistence

## Project Structure

```
emailcls/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── spam_classifier.pth
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessor.py
│   ├── models/
│   │   └── classifier.py
│   ├── training/
│   │   └── trainer.py
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── utils/
│   │   └── config.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Requirements

All required packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Usage

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place your training data in `data/train.csv`
   - Place your test data in `data/test.csv`
   - Each CSV file should have two columns: 'text' (email content) and 'label' (0 for ham, 1 for spam)

3. Run the classifier:
```bash
python src/main.py

python -m src.main
```

## Model Architecture

The neural network consists of:
- Input layer (TF-IDF features)
- Hidden layer 1 (256 neurons) with batch normalization
- Hidden layer 2 (128 neurons) with batch normalization
- Output layer (2 neurons for binary classification)
- ReLU activation functions
- Dropout layers (0.5) for regularization

## Features

### Text Preprocessing
- Lowercase conversion
- Special character removal
- Tokenization
- Stopword removal
- TF-IDF vectorization with n-grams

### Training
- Batch processing
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Progress tracking with tqdm

### Evaluation
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Training history plots (loss and accuracy)
- Support for both training and validation metrics

## Output

The script will generate:
1. Trained model saved in `models/spam_classifier.pth`
2. TF-IDF vectorizer saved in `models/tfidf_vectorizer.pkl`
3. Confusion matrix plot saved as `confusion_matrix.png`
4. Training history plots saved as `training_history.png`
5. Classification report printed to console

## Customization

You can modify the following parameters in `src/utils/config.py`:
- `MAX_FEATURES`: Number of TF-IDF features (default: 5000)
- `HIDDEN_SIZE_1`: Size of first hidden layer (default: 256)
- `HIDDEN_SIZE_2`: Size of second hidden layer (default: 128)
- `DROPOUT_RATE`: Dropout rate (default: 0.5)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `NUM_EPOCHS`: Number of training epochs (default: 10)
- `TEST_SIZE`: Validation set size (default: 0.2)

## Sample Data

The project includes sample training and test datasets in the `data` directory. These can be used to test the implementation or as templates for your own data 
