import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from ..utils.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DEVICE
from ..models.classifier import SpamClassifier
from ..data.dataset import EmailDataset

class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None):
        self.model = model.to(DEVICE)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        ) if val_dataset else None
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Early stopping parameters
        self.patience = 5
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Gradient clipping
        self.max_grad_norm = 1.0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in tqdm(self.train_loader, desc="Training"):
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Add L2 regularization
            l2_loss = self.model.get_l2_loss()
            loss += l2_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        """Validate the model."""
        if not self.val_loader:
            return None, None
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(self.val_loader, desc="Validation"):
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(self.val_loader), correct / total
    
    def train(self):
        """Train the model for multiple epochs."""
        best_val_acc = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            if self.val_loader:
                val_loss, val_acc = self.validate()
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.model.save_model()
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        break
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        } 