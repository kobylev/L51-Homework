import matplotlib.pyplot as plt
import numpy as np
from src.data.dataset import generate_signal
import torch

def build_noise_histogram(config) -> plt.Figure:
    def get_noise(seed):
        np.random.seed(seed)
        amps = np.random.uniform(0.8, 1.2, (1000, 4))
        phases = np.random.uniform(0, 2 * np.pi, (1000, 4))
        return amps.flatten(), phases.flatten()

    train_amps, train_phases = get_noise(42)
    test_amps, test_phases = get_noise(123)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.hist(train_amps, alpha=0.5, label='Train Amps', bins=20)
    ax1.hist(test_amps, alpha=0.5, label='Test Amps', bins=20)
    ax1.set_title("Amplitude Noise Distribution")
    ax1.legend()
    
    ax2.hist(train_phases, alpha=0.5, label='Train Phases', bins=20)
    ax2.hist(test_phases, alpha=0.5, label='Test Phases', bins=20)
    ax2.set_title("Phase Noise Distribution")
    ax2.legend()
    return fig

def build_prediction_plot(model, config, freq_idx) -> plt.Figure:
    mixed_test, clean_test, t = generate_signal(config.FREQUENCIES, config.SAMPLE_RATE, 1, noise_seed=999)
    control = np.zeros(len(config.FREQUENCIES))
    control[freq_idx] = 1.0
    input_data = np.zeros((1, config.SAMPLE_RATE, 1 + len(config.FREQUENCIES)))
    input_data[0, :, 0] = mixed_test
    input_data[0, :, 1:] = control
    
    x = torch.tensor(input_data, dtype=torch.float32).to(config.DEVICE)
    with torch.no_grad():
        pred, _ = model(x)
    
    pred = pred.squeeze().cpu().numpy()
    target = clean_test[freq_idx]
    
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t[:200], target[:200], label='True Clean Signal')
    plt.plot(t[:200], pred[:200], '--', label='LSTM Prediction')
    plt.title(f"Frequency Extraction: {config.FREQUENCIES[freq_idx]}Hz")
    plt.legend()
    return fig

def build_ablation_plot(ablation_results) -> plt.Figure:
    t = ablation_results['t']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(t[:200], ablation_results['clean'][0][:200], label='True 1Hz')
    ax1.plot(t[:200], ablation_results['pred_1Hz'][:200], '--', label='Ablated Pred 1Hz')
    ax1.set_title("Ablation Effect on 1Hz")
    ax1.legend()
    
    ax2.plot(t[:200], ablation_results['clean'][3][:200], label='True 7Hz')
    ax2.plot(t[:200], ablation_results['pred_7Hz'][:200], '--', label='Ablated Pred 7Hz')
    ax2.set_title("Ablation Effect on 7Hz")
    ax2.legend()
    return fig

def build_comparison_grid(model_l1, model_l100, config) -> plt.Figure:
    mixed_test, clean_test, _ = generate_signal(config.FREQUENCIES, config.SAMPLE_RATE, 1, noise_seed=123)
    num_samples = 1000
    mixed_input = mixed_test[:num_samples]
    t_axis = np.arange(num_samples)
    
    preds_l1 = []
    preds_l100 = []
    
    for i in range(len(config.FREQUENCIES)):
        control = np.zeros(len(config.FREQUENCIES))
        control[i] = 1.0
        input_data = np.zeros((1, num_samples, 1 + len(config.FREQUENCIES)))
        input_data[0, :, 0] = mixed_input
        input_data[0, :, 1:] = control
        x = torch.tensor(input_data, dtype=torch.float32).to(config.DEVICE)
        
        with torch.no_grad():
            # L=1 inference
            p_l1 = []
            for t in range(num_samples):
                out, _ = model_l1(x[:, t:t+1, :], None)
                p_l1.append(out.item())
            preds_l1.append(np.array(p_l1))
            
            # L=100 inference
            p_l100, _ = model_l100(x)
            preds_l100.append(p_l100.squeeze().cpu().numpy())
            
    fig, axes = plt.subplots(5, 3, figsize=(14, 18), tight_layout=True)
    rows = ["Mixed"] + [f"{f} Hz" for f in config.FREQUENCIES]
    cols = ["Ground Truth", "LSTM  L=1", "LSTM  L=100"]
    
    for r in range(5):
        for c in range(3):
            ax = axes[r, c]
            if r == 0:
                ax.plot(t_axis, mixed_input, linewidth=0.8)
            else:
                f_idx = r - 1
                if c == 0: data = clean_test[f_idx, :num_samples]
                elif c == 1: data = preds_l1[f_idx]
                else: data = preds_l100[f_idx]
                ax.plot(t_axis, data, linewidth=0.8)
            if c == 0: ax.set_ylabel(rows[r])
            if r == 0: ax.set_title(cols[c])
            ax.set_xlabel("Sample")
    return fig
