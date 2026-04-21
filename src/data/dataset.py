import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def generate_signal(freqs, sample_rate, duration, noise_seed=None):
    if noise_seed is not None:
        np.random.seed(noise_seed)
    
    t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    signals = []
    clean_signals = []
    
    for f in freqs:
        amp = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        
        clean_sig = np.sin(2 * np.pi * f * t)
        noisy_sig = amp * np.sin(2 * np.pi * f * t + phase)
        
        signals.append(noisy_sig)
        clean_signals.append(clean_sig)
        
    mixed_signal = np.sum(signals, axis=0) / 4.0
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
        
        mixed_window = self.mixed_signal[sample_start : sample_start + self.window_size]
        
        control = np.zeros(self.num_freqs)
        control[freq_idx] = 1.0
        
        input_data = np.zeros((self.window_size, 1 + self.num_freqs))
        input_data[:, 0] = mixed_window
        input_data[:, 1:] = control
        
        label = self.clean_signals[freq_idx, sample_start : sample_start + self.window_size]
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).unsqueeze(-1)

def get_dataloaders(config):
    train_samples = int(config.NUM_SAMPLES * config.TRAIN_SPLIT)
    
    mixed_train, clean_train, _ = generate_signal(config.FREQUENCIES, config.SAMPLE_RATE, config.DURATION, noise_seed=42)
    mixed_test, clean_test, _ = generate_signal(config.FREQUENCIES, config.SAMPLE_RATE, config.DURATION, noise_seed=123)
    
    train_ds = BandpassDataset(mixed_train[:train_samples], clean_train[:, :train_samples], config.CONTEXT_WINDOW)
    test_ds = BandpassDataset(mixed_test[train_samples:], clean_test[:, train_samples:], config.CONTEXT_WINDOW)
    
    shuffle_train = (config.L_PARAMETER == 1)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=shuffle_train)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader
