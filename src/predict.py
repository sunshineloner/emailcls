import torch
from src.data.preprocessor import TextPreprocessor
from src.models.classifier import SpamClassifier
from src.utils.config import MODEL_SAVE_PATH, DEVICE

def predict_spam(text):
    """
    Predict if the given text is spam or not.
    
    Args:
        text (str): The text to classify
        
    Returns:
        tuple: (prediction (0 or 1), confidence score)
    """
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Load the vectorizer
    if not preprocessor.load_vectorizer():
        print("Error: No trained vectorizer found. Please train the model first.")
        return None, None
    
    # Clean and transform the text
    cleaned_text = preprocessor.clean_text(text)
    text_features = preprocessor.transform([cleaned_text])
    
    # Convert to tensor
    text_tensor = torch.FloatTensor(text_features.toarray())
    
    # Initialize model
    input_size = text_tensor.shape[1]
    model = SpamClassifier(input_size)
    
    # Load trained weights
    if not model.load_model():
        print("Error: No trained model found. Please train the model first.")
        return None, None
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        text_tensor = text_tensor.to(DEVICE)
        outputs = model(text_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

def main():
    print("\n=== Email Spam Classifier ===")
    print("Enter 'quit' to exit")
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

if __name__ == "__main__":
    main() 