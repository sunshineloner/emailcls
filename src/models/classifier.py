import torch
import torch.nn as nn
from ..utils.config import (
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
    OUTPUT_SIZE,
    DROPOUT_RATE,
    MODEL_SAVE_PATH
)

class SpamClassifier(nn.Module):
    def __init__(self, input_size):
        """
        Initialize the spam classifier.
        
        Args:
            input_size: Size of the input features
        """
        super(SpamClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, HIDDEN_SIZE_1)
        self.layer2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.layer3 = nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.batch_norm1 = nn.BatchNorm1d(HIDDEN_SIZE_1)
        self.batch_norm2 = nn.BatchNorm1d(HIDDEN_SIZE_2)
        self.l2_reg = 0.01  # L2 regularization factor
    
    def forward(self, x):
        # First layer
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.layer3(x)
        return x
    
    def get_l2_loss(self):
        """Calculate L2 regularization loss."""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_reg * l2_loss
    
    def save_model(self):
        """Save the model to disk."""
        torch.save(self.state_dict(), MODEL_SAVE_PATH)
    
    def load_model(self):
        """Load the model from disk."""
        if MODEL_SAVE_PATH.exists():
            self.load_state_dict(torch.load(MODEL_SAVE_PATH))
            return True
        return False 