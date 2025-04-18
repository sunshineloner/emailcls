import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.config import (
    TRAIN_FILE,
    TEST_FILE,
    TEST_SIZE,
    RANDOM_STATE,
    DEVICE
)
from src.data.preprocessor import TextPreprocessor, load_data
from src.data.dataset import EmailDataset
from src.models.classifier import SpamClassifier
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.predict import predict_spam

def train_model():
    """Train the spam classifier model."""
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    # Check if CUDA is available
    print(f"Using device: {DEVICE}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_data(TRAIN_FILE, TEST_FILE)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Transform data
    print("Transforming data...")
    X_train_tfidf = preprocessor.fit_transform(X_train)
    X_test_tfidf = preprocessor.transform(X_test)
    
    # Save the vectorizer
    preprocessor.save_vectorizer()
    
    # Split training data into train and validation sets
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_tfidf, y_train,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # Create datasets
    train_dataset = EmailDataset(X_train_split, y_train_split)
    val_dataset = EmailDataset(X_val_split, y_val_split)
    test_dataset = EmailDataset(X_test_tfidf, y_test)
    
    # Initialize model with correct input size
    print("\nInitializing model...")
    input_size = X_train_tfidf.shape[1]  # Get the actual number of features
    model = SpamClassifier(input_size)
    
    # Train model
    print("\nTraining model...")
    trainer = Trainer(model, train_dataset, val_dataset)
    history = trainer.train()
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluator = Evaluator(model, test_dataset)
    evaluator.generate_report()
    evaluator.plot_training_history(history)
    
    print("\nTraining completed! Check the generated plots and reports.")

def prediction_mode():
    """Run the spam prediction mode."""
    print("\n=== Email Spam Classifier - Prediction Mode ===")
    print("Enter 'quit' to return to main menu")
    print("Enter 'feedback' to provide feedback")
    
    while True:
        print("\nEnter email text:")
        text = input("> ").strip()
        
        if text.lower() == 'quit':
            break
        elif text.lower() == 'feedback':
            print("\nPlease provide your feedback:")
            feedback = input("> ").strip()
            print("Thank you for your feedback!")
            continue
        
        if not text:
            print("Please enter some text!")
            continue
        
        # Get prediction
        prediction, confidence = predict_spam(text)
        
        if prediction is not None:
            result = "SPAM" if prediction == 1 else "NOT SPAM"
            print(f"\nPrediction: {result}")
            print(f"Confidence: {confidence:.2%}")
            
            # Ask for feedback
            print("\nWas this prediction correct? (y/n)")
            feedback = input("> ").strip().lower()
            if feedback == 'y':
                print("Thank you for your feedback!")
            elif feedback == 'n':
                print("Thank you for helping improve the model!")

def main():
    while True:
        print("\n=== Email Spam Classifier ===")
        print("1. Train Model")
        print("2. Prediction Mode")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            prediction_mode()
        elif choice == '3':
            print("\nThank you for using Email Spam Classifier!")
            break
        else:
            print("\nInvalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 