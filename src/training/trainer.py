import torch
import torch.nn as nn
from src.data.dataset import get_dataloaders
from src.models.model import ConditionalLSTM

def train_one_seed(config, seed):
    torch.manual_seed(seed)
    train_loader, test_loader = get_dataloaders(config)
    
    model = ConditionalLSTM(
        config.INPUT_SIZE, 
        config.HIDDEN_SIZE, 
        config.NUM_LAYERS, 
        config.OUTPUT_SIZE
    ).to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_mse = float('inf')
    
    for epoch in range(config.EPOCHS):
        model.train()
        hidden = None
        for x, y in train_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            
            if config.L_PARAMETER == 1:
                hidden = None
            else:
                if hidden is not None:
                    if hidden[0].size(1) != x.size(0):
                        hidden = None
                    else:
                        hidden = (hidden[0].detach(), hidden[1].detach())
            
            output, hidden = model(x, hidden)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
    return model
