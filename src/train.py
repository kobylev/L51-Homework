import torch
import torch.nn as nn
from src.config import Config

def train_one_epoch(model, loader, optimizer, criterion, device, L_parameter):
    model.train()
    total_loss = 0
    hidden = None
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        if L_parameter == 1:
            hidden = None
        else:
            if hidden is not None:
                # Check if batch size changed (e.g., last batch)
                if hidden[0].size(1) != x.size(0):
                    hidden = None
                else:
                    hidden = (hidden[0].detach(), hidden[1].detach())
        
        output, hidden = model(x, hidden)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, L_parameter):
    model.eval()
    total_loss = 0
    hidden = None
    predictions = []
    targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            if L_parameter == 1:
                hidden = None
            else:
                if hidden is not None:
                    if hidden[0].size(1) != x.size(0):
                        hidden = None
            
            output, hidden = model(x, hidden)
            loss = criterion(output, y)
            total_loss += loss.item()
            
            predictions.append(output.cpu())
            targets.append(y.cpu())
            
    return total_loss / len(loader), torch.cat(predictions), torch.cat(targets)
