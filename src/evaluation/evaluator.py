import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from ..utils.config import DEVICE
from ..data.dataset import EmailDataset

class Evaluator:
    def __init__(self, model, test_dataset):
        self.model = model.to(DEVICE)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
    
    def evaluate(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(self.test_loader, desc="Evaluating"):
                batch_X = batch_X.to(DEVICE)
                outputs = self.model(batch_X)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        return all_preds, all_labels
    
    def generate_report(self):
        """Generate classification report and confusion matrix."""
        y_pred, y_true = self.evaluate()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def plot_training_history(self, history):
        """Plot training and validation metrics."""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Train Loss')
        if history['val_losses']:
            plt.plot(history['val_losses'], label='Val Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accs'], label='Train Acc')
        if history['val_accs']:
            plt.plot(history['val_accs'], label='Val Acc')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close() 