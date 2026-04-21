import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import Config

def generate_signal(freqs, sample_rate, duration, noise_seed=None):
    if noise_seed is not None:
        np.random.seed(noise_seed)
    
    t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    signals = []
    clean_signals = []
    
    for f in freqs:
        # Noise injection: Amplitude (0.8 to 1.2) and Phase (0 to 2π)
        amp = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Clean signal for label (base amplitude 1, no random phase for simplicity in label?)
        # User said: "The label (Y) is the clean (noiseless) context window of the requested frequency."
        # Usually "clean" means amplitude 1 and phase 0 unless otherwise specified.
        clean_sig = np.sin(2 * np.pi * f * t)
        
        # Noisy signal
        noisy_sig = amp * np.sin(2 * np.pi * f * t + phase)
        
        signals.append(noisy_sig)
        clean_signals.append(clean_sig)
        
    mixed_signal = np.sum(signals, axis=0) / 4.0 # Normalize by 4
    return mixed_signal, np.array(clean_signals), t

class BandpassDataset(Dataset):
    def __init__(self, mixed_signal, clean_signals, window_size):
        self.mixed_signal = mixed_signal
        self.clean_signals = clean_signals
        self.window_size = window_size
        self.num_freqs = clean_signals.shape[0]
        self.num_samples = len(mixed_signal) - window_size
        
    def __len__(self):
        return self.num_samples * self.num_freqs
    
    def __getitem__(self, idx):
        freq_idx = idx % self.num_freqs
        sample_start = idx // self.num_freqs
        
        # Mixed signal window
        mixed_window = self.mixed_signal[sample_start : sample_start + self.window_size]
        
        # Control vector (One-hot)
        control = np.zeros(self.num_freqs)
        control[freq_idx] = 1.0
        
        # Input: [mixed_signal, control...] repeated for each time step? 
        # Or just append control to each time step?
        # Typically LSTM input is (seq_len, input_size)
        # So mixed_window (seq_len, 1) + control (4,) broadcasted
        
        input_data = np.zeros((self.window_size, 1 + self.num_freqs))
        input_data[:, 0] = mixed_window
        input_data[:, 1:] = control
        
        # Label: clean signal window
        label = self.clean_signals[freq_idx, sample_start : sample_start + self.window_size]
        
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).unsqueeze(-1)

def get_dataloaders(window_size, batch_size, L_parameter):
    # Train set (80%)
    train_samples = int(Config.NUM_SAMPLES * Config.TRAIN_SPLIT)
    
    # Generate signals with different seeds
    mixed_train, clean_train, _ = generate_signal(Config.FREQUENCIES, Config.SAMPLE_RATE, Config.DURATION, noise_seed=42)
    mixed_test, clean_test, _ = generate_signal(Config.FREQUENCIES, Config.SAMPLE_RATE, Config.DURATION, noise_seed=123)
    
    # Split them manually to ensure 80/20 of the 10s duration?
    # Actually, the user says "Strictly split the data into 80% Train / 20% Test. YOU MUST generate independent noise realizations"
    # This might mean 8s train, 2s test? Or separate 10s signals.
    # "80% Train / 20% Test" usually refers to the total dataset size.
    # Let's generate 10s for train and 10s for test but only use 80% and 20% of the possible windows?
    # Or just use the first 8000 samples for train and last 2000 for test from different signal realizations.
    
    train_ds = BandpassDataset(mixed_train[:train_samples], clean_train[:, :train_samples], window_size)
    test_ds = BandpassDataset(mixed_test[train_samples:], clean_test[:, train_samples:], window_size)
    
    shuffle_train = (L_parameter == 1)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
