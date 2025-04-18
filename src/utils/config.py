from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Data parameters
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model parameters
INPUT_SIZE = MAX_FEATURES
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
OUTPUT_SIZE = 2
DROPOUT_RATE = 0.7

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
NUM_EPOCHS = 20
DEVICE = "cpu"  # Using CPU instead of CUDA

# Model saving
MODEL_SAVE_PATH = MODEL_DIR / "spam_classifier.pth"
VECTORIZER_SAVE_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True) 