import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.data.dataset import get_dataloaders, generate_signal

def run_evaluation(model, config):
    _, test_loader = get_dataloaders(config)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    hidden = None
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            if config.L_PARAMETER == 1:
                hidden = None
            else:
                if hidden is not None and hidden[0].size(1) != x.size(0):
                    hidden = None
            
            output, hidden = model(x, hidden)
            loss = criterion(output, y)
            total_loss += loss.item()
            
    return {"mse": total_loss / len(test_loader)}

def get_ablation_results(model, config):
    model.eval()
    mixed_test, clean_test, t = generate_signal(config.FREQUENCIES, config.SAMPLE_RATE, 1, noise_seed=999)
    
    # Identify top-K units for 1Hz
    control = np.zeros(len(config.FREQUENCIES))
    control[0] = 1.0
    input_data = np.zeros((1, config.SAMPLE_RATE, 1 + len(config.FREQUENCIES)))
    input_data[0, :, 0] = mixed_test
    input_data[0, :, 1:] = control
    x = torch.tensor(input_data, dtype=torch.float32).to(config.DEVICE)
    
    activations = []
    def hook(module, input, output):
        activations.append(output[0].detach())

    handle = model.lstm.register_forward_hook(hook)
    model(x)
    handle.remove()
    
    avg_activations = torch.abs(activations[0]).mean(dim=(0, 1))
    top_k_indices = torch.topk(avg_activations, k=10).indices
    
    # Prune and test
    original_weights = model.fc.weight.data.clone()
    model.fc.weight.data[:, top_k_indices] = 0.0
    
    results = {"t": t, "clean": clean_test}
    for i in [0, 3]: # 1Hz and 7Hz
        f = config.FREQUENCIES[i]
        control_vec = np.zeros(len(config.FREQUENCIES))
        control_vec[i] = 1.0
        input_data[0, :, 1:] = control_vec
        x_in = torch.tensor(input_data, dtype=torch.float32).to(config.DEVICE)
        with torch.no_grad():
            pred, _ = model(x_in)
        results[f"pred_{f}Hz"] = pred.squeeze().cpu().numpy()
        
    model.fc.weight.data = original_weights
    return results
