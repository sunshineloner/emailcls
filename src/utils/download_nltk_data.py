import nltk

def download_nltk_data():
    """Download required NLTK data."""
    required_data = [
        'punkt',
        'stopwords',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'wordnet'
    ]
    
    print("Downloading required NLTK data...")
    for data in required_data:
        try:
            nltk.download(data, quiet=True)
            print(f"Successfully downloaded {data}")
        except Exception as e:
            print(f"Error downloading {data}: {str(e)}")
    
    print("\nNLTK data download completed!")

if __name__ == "__main__":
    download_nltk_data() 