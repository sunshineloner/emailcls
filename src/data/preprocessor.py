import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from pathlib import Path
from ..utils.config import VECTORIZER_SAVE_PATH, MAX_FEATURES

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
    
    def clean_text(self, text):
        """Clean and preprocess the input text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Remove short words (less than 2 characters)
        tokens = [token for token in tokens if len(token) > 1]
        
        return ' '.join(tokens)
    
    def fit_transform(self, texts):
        """Fit the vectorizer and transform the texts."""
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.fit_transform(cleaned_texts)
    
    def transform(self, texts):
        """Transform new texts using the fitted vectorizer."""
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts)
    
    def save_vectorizer(self):
        """Save the fitted vectorizer to disk."""
        with open(VECTORIZER_SAVE_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_vectorizer(self):
        """Load the fitted vectorizer from disk."""
        if VECTORIZER_SAVE_PATH.exists():
            with open(VECTORIZER_SAVE_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
            return True
        return False

def load_data(train_path, test_path):
    """Load and prepare the dataset."""
    # Load training data
    train_data = pd.read_csv(train_path)
    X_train = train_data['text'].values
    y_train = train_data['label'].values
    
    # Load test data
    test_data = pd.read_csv(test_path)
    X_test = test_data['text'].values
    y_test = test_data['label'].values
    
    return X_train, X_test, y_train, y_test 