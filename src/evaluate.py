import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import Config
from src.datasets import generate_signal, BandpassDataset
from torch.utils.data import DataLoader

def plot_noise_histogram(docs_dir):
    # Generate independent noise realizations
    _, _, _ = generate_signal(Config.FREQUENCIES, Config.SAMPLE_RATE, Config.DURATION, noise_seed=42)
    # We need to capture the actual noise values.
    # Let's modify generate_signal or just re-simulate the noise part.
    
    def get_noise(seed):
        np.random.seed(seed)
        amps = np.random.uniform(0.8, 1.2, (1000, 4))
        phases = np.random.uniform(0, 2 * np.pi, (1000, 4))
        return amps.flatten(), phases.flatten()

    train_amps, train_phases = get_noise(42)
    test_amps, test_phases = get_noise(123)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(train_amps, alpha=0.5, label='Train Amps', bins=20)
    plt.hist(test_amps, alpha=0.5, label='Test Amps', bins=20)
    plt.title("Amplitude Noise Distribution")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(train_phases, alpha=0.5, label='Train Phases', bins=20)
    plt.hist(test_phases, alpha=0.5, label='Test Phases', bins=20)
    plt.title("Phase Noise Distribution")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "noise_histogram.png"))
    plt.close()

def plot_predictions(model, device, docs_dir):
    model.eval()
    freqs = Config.FREQUENCIES
    # Generate a fresh test signal
    mixed_test, clean_test, t = generate_signal(freqs, Config.SAMPLE_RATE, 1, noise_seed=999) # 1 sec for plotting
    
    for i, f in enumerate(freqs):
        # Create a single sequence for this frequency
        control = np.zeros(len(freqs))
        control[i] = 1.0
        
        input_data = np.zeros((1, Config.SAMPLE_RATE, 1 + len(freqs)))
        input_data[0, :, 0] = mixed_test
        input_data[0, :, 1:] = control
        
        x = torch.tensor(input_data, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred, _ = model(x)
        
        pred = pred.squeeze().cpu().numpy()
        target = clean_test[i]
        
        plt.figure(figsize=(10, 4))
        plt.plot(t[:200], target[:200], label='True Clean Signal')
        plt.plot(t[:200], pred[:200], '--', label='LSTM Prediction')
        plt.title(f"Frequency Extraction: {f}Hz")
        plt.legend()
        plt.savefig(os.path.join(docs_dir, f"prediction_{f}Hz.png"))
        plt.close()

def perform_targeted_ablation(model, device, docs_dir):
    model.eval()
    # 1. Identify top-K hidden units for 1Hz
    # We'll use the 1Hz extraction task and see which hidden units are most active
    freqs = Config.FREQUENCIES
    mixed_test, _, _ = generate_signal(freqs, Config.SAMPLE_RATE, 1, noise_seed=999)
    
    control = np.zeros(len(freqs))
    control[0] = 1.0 # 1Hz
    
    input_data = np.zeros((1, Config.SAMPLE_RATE, 1 + len(freqs)))
    input_data[0, :, 0] = mixed_test
    input_data[0, :, 1:] = control
    
    x = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    # Hook to get LSTM output (hidden states for each time step)
    activations = []
    def hook(module, input, output):
        # output is (out, (h, c))
        # out is (batch, seq, hidden_size)
        activations.append(output[0].detach())

    handle = model.lstm.register_forward_hook(hook)
    model(x)
    handle.remove()
    
    # activations[0] is (1, 1000, 64)
    avg_activations = torch.abs(activations[0]).mean(dim=(0, 1)) # (64,)
    top_k_indices = torch.topk(avg_activations, k=10).indices
    
    # 2. Zero out these specific weights in the FC layer (or LSTM output)
    # Easiest is to zero out the FC layer weights corresponding to these hidden units
    original_weights = model.fc.weight.data.clone()
    model.fc.weight.data[:, top_k_indices] = 0.0
    
    # 3. Test extraction for 1Hz and 7Hz
    results = {}
    for i in [0, 3]: # 1Hz and 7Hz
        f = freqs[i]
        control = np.zeros(len(freqs))
        control[i] = 1.0
        input_data[0, :, 1:] = control
        x = torch.tensor(input_data, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred, _ = model(x)
        results[f] = pred.squeeze().cpu().numpy()
    
    # Restore weights
    model.fc.weight.data = original_weights
    
    # Plot ablation results
    _, clean_test, t = generate_signal(freqs, Config.SAMPLE_RATE, 1, noise_seed=999)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t[:200], clean_test[0][:200], label='True 1Hz')
    plt.plot(t[:200], results[1][:200], '--', label='Ablated Pred 1Hz')
    plt.title("Ablation Effect on 1Hz")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(t[:200], clean_test[3][:200], label='True 7Hz')
    plt.plot(t[:200], results[7][:200], '--', label='Ablated Pred 7Hz')
    plt.title("Ablation Effect on 7Hz")
    plt.legend()
    
    plt.savefig(os.path.join(docs_dir, "ablation_plot.png"))
    plt.close()

def plot_window_size_ablation(results_10, results_100, docs_dir):
    # results_10 and 100 are lists of MSEs
    labels = ['Window=10', 'Window=100']
    means = [np.mean(results_10), np.mean(results_100)]
    stds = [np.std(results_10), np.std(results_100)]
    
    plt.figure(figsize=(6, 5))
    plt.bar(labels, means, yerr=stds, capsize=10, color=['blue', 'green'], alpha=0.7)
    plt.ylabel("Mean Squared Error")
    plt.title("Window Size Ablation")
    plt.savefig(os.path.join(docs_dir, "window_size_ablation.png"))
    plt.close()
