import torch
import torch.nn as nn
import numpy as np
import os
from src.config import Config
from src.datasets import get_dataloaders
from src.model import ConditionalLSTM
from src.train import train_one_epoch, evaluate
from src.evaluate import (
    plot_noise_histogram, 
    plot_predictions, 
    perform_targeted_ablation, 
    plot_window_size_ablation
)

def run_experiment(seed, window_size, L_parameter):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader, test_loader = get_dataloaders(window_size, Config.BATCH_SIZE, L_parameter)
    
    model = ConditionalLSTM(
        Config.INPUT_SIZE, 
        Config.HIDDEN_SIZE, 
        Config.NUM_LAYERS, 
        Config.OUTPUT_SIZE
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_mse = float('inf')
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, L_parameter)
        test_mse, _, _ = evaluate(model, test_loader, criterion, Config.DEVICE, L_parameter)
        if test_mse < best_mse:
            best_mse = test_mse
            
    return best_mse, model

def main():
    if not os.path.exists(Config.DOCS_DIR):
        os.makedirs(Config.DOCS_DIR)
        
    print("Starting Experiments...")
    
    # 1. Noise Independence Proof
    plot_noise_histogram(Config.DOCS_DIR)
    
    # 2. Statistical Aggregation (5 seeds, L=1 vs L=100)
    results_L1 = []
    results_L100 = []
    
    for seed in range(Config.NUM_SEEDS):
        print(f"Seed {seed}: Training L=1...")
        mse_l1, model_l1 = run_experiment(seed, Config.CONTEXT_WINDOW, 1)
        results_L1.append(mse_l1)
        
        print(f"Seed {seed}: Training L=100...")
        mse_l100, _ = run_experiment(seed, Config.CONTEXT_WINDOW, 100)
        results_L100.append(mse_l100)
        
        # Save plots from the first seed's L=1 model
        if seed == 0:
            print("Generating Prediction Plots...")
            plot_predictions(model_l1, Config.DEVICE, Config.DOCS_DIR)
            print("Performing Targeted Ablation...")
            perform_targeted_ablation(model_l1, Config.DEVICE, Config.DOCS_DIR)
            
    # 3. Window Size Ablation (using L=1 for comparison)
    print("Running Window Size Ablation (Window=10)...")
    results_W10 = []
    for seed in range(Config.NUM_SEEDS):
        mse_w10, _ = run_experiment(seed, 10, 1)
        results_W10.append(mse_w10)
    
    plot_window_size_ablation(results_W10, results_L1, Config.DOCS_DIR)
    
    # Print Summary Table (to be used in README)
    print("\n--- RESULTS SUMMARY ---")
    print(f"L=1: Mean MSE = {np.mean(results_L1):.6f} +/- {np.std(results_L1):.6f}")
    print(f"L=100: Mean MSE = {np.mean(results_L100):.6f} +/- {np.std(results_L100):.6f}")
    print(f"Window=10: Mean MSE = {np.mean(results_W10):.6f} +/- {np.std(results_W10):.6f}")
    print("------------------------")

if __name__ == "__main__":
    main()
