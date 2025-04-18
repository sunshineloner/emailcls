import torch
from torch.utils.data import Dataset

class EmailDataset(Dataset):
    def __init__(self, X, y):
        """
        Initialize the dataset.
        
        Args:
            X: Feature matrix (numpy array or sparse matrix)
            y: Target labels (numpy array)
        """
        self.X = torch.FloatTensor(X.toarray() if hasattr(X, 'toarray') else X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.y)
    
    def __getitem__(self, idx):
        """Get a sample by index."""
        return self.X[idx], self.y[idx] 