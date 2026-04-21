import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import Config
from src.datasets import generate_signal
from src.model import ConditionalLSTM

def main():
    # 1. Load trained models
    device = torch.device("cpu") # Inference on CPU for simplicity
    
    model_l1 = ConditionalLSTM(
        Config.INPUT_SIZE, Config.HIDDEN_SIZE, Config.NUM_LAYERS, Config.OUTPUT_SIZE
    )
    model_l1.load_state_dict(torch.load(Config.MODEL_L1_PATH, map_location=device))
    model_l1.eval()
    
    model_l100 = ConditionalLSTM(
        Config.INPUT_SIZE, Config.HIDDEN_SIZE, Config.NUM_LAYERS, Config.OUTPUT_SIZE
    )
    model_l100.load_state_dict(torch.load(Config.MODEL_L100_PATH, map_location=device))
    model_l100.eval()
    
    # 2. Generate test data
    # Use seed 123 as specified for test set
    mixed_test, clean_test, _ = generate_signal(
        Config.FREQUENCIES, Config.SAMPLE_RATE, Config.DURATION, noise_seed=123
    )
    
    num_samples = 1000
    mixed_input = mixed_test[:num_samples]
    t_axis = np.arange(num_samples)
    
    # 3. Perform Inference
    predictions_l1 = []
    predictions_l100 = []
    
    for i, freq in enumerate(Config.FREQUENCIES):
        control = np.zeros(len(Config.FREQUENCIES))
        control[i] = 1.0
        
        # Prepare input tensor (1, 1000, 5)
        input_data = np.zeros((1, num_samples, 1 + len(Config.FREQUENCIES)))
        input_data[0, :, 0] = mixed_input
        input_data[0, :, 1:] = control
        x_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        with torch.no_grad():
            # L=1: step sample-by-sample with hidden reset
            # However, our model is trained for sequence. 
            # Per prompt: "For L=1: reset hidden state to zeros at t=0, then step sample-by-sample"
            # This implies feeding 1 sample at a time.
            preds_step = []
            hidden = None
            for t in range(num_samples):
                out, hidden = model_l1(x_tensor[:, t:t+1, :], None) # Reset hidden every step?
                # Prompt says: "reset hidden state to zeros at t=0, then step sample-by-sample"
                # "L=1: Reset the hidden state after EVERY context window" in Phase 1.
                # So for L=1 inference, we should reset hidden state every step.
                preds_step.append(out.item())
            predictions_l1.append(np.array(preds_step))
            
            # L=100: full 1000-sample sequence in one forward pass
            pred_full, _ = model_l100(x_tensor)
            predictions_l100.append(pred_full.squeeze().numpy())
            
    # 4. Build the figure
    fig, axes = plt.subplots(5, 3, figsize=(14, 18), tight_layout=True)
    fig.suptitle("Results — Run 1: L=1 vs L=100 (freq_noise = 0.0)", fontsize=16)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    rows = ["Mixed"] + [f"{f} Hz" for f in Config.FREQUENCIES]
    cols = ["Ground Truth", "LSTM  L=1", "LSTM  L=100"]
    
    for r in range(5):
        for c in range(3):
            ax = axes[r, c]
            
            if r == 0:
                # Mixed row
                data = mixed_input
                ax.plot(t_axis, data, color='C0', linewidth=0.8)
            else:
                freq_idx = r - 1
                if c == 0:
                    # Ground Truth
                    data = clean_test[freq_idx, :num_samples]
                elif c == 1:
                    # L=1
                    data = predictions_l1[freq_idx]
                else:
                    # L=100
                    data = predictions_l100[freq_idx]
                
                ax.plot(t_axis, data, color='C0', linewidth=0.8)
            
            # Labels and titles
            if c == 0:
                ax.set_ylabel(rows[r], fontsize=12)
            
            if r == 0:
                ax.set_title(cols[c], fontsize=14)
            
            ax.set_xlabel("Sample")
            ax.grid(False)
            
    # Save the figure
    save_path = os.path.join(Config.PLOTS_DIR, "comparison_grid_run1.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Saved comparison grid to {save_path}")

if __name__ == "__main__":
    main()
