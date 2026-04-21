# Conditional LSTM Bandpass Filter

## Overview
This project implements a **Conditional LSTM Bandpass Filter** using PyTorch. The model is designed to extract specific frequency components (1Hz, 3Hz, 5Hz, or 7Hz) from a normalized mixture of noisy signals, controlled by a one-hot input vector.

## The Core Idea
The fundamental intuition is that a Recurrent Neural Network (LSTM) can be conditioned to act as a dynamic bandpass filter. By appending a control vector $C$ to the input signal $X(t)$, the network learns to attend to specific temporal patterns corresponding to the target frequency.

### Mathematical Synthesis
Signals are generated as:
$$S_f(t) = A_f \sin(2\pi f t + \phi_f)$$
where $A_f \in [0.8, 1.2]$ and $\phi_f \in [0, 2\pi]$ are independent noise realizations. The mixed signal is:
$$X(t) = \frac{1}{4} \sum_{f \in \{1,3,5,7\}} S_f(t)$$
The LSTM model $f_\theta(X_{t-k:t}, C)$ predicts the clean signal $Y_f(t)$ for the frequency specified by $C$.

## Project Structure
```text
C:\Ai_Expert\L51-Homework\
├── docs/                      # Result visualizations and plots
│   ├── ablation_plot.png      # Targeted pruning results
│   ├── noise_histogram.png    # Train/Test noise independence
│   ├── prediction_1Hz.png     # Frequency extraction plots
│   ├── ...
│   └── window_size_ablation.png
├── src/                       # Source code
│   ├── config.py              # Hyperparameters
│   ├── datasets.py            # Signal synthesis and data loading
│   ├── evaluate.py            # Plotting and ablation logic
│   ├── main.py                # Orchestration
│   ├── model.py               # LSTM architecture
│   └── train.py               # Training loops
├── requirements.txt           # Dependencies
└── README.md                  # Academic documentation
```

## Architecture
Data flows through a single-layer LSTM followed by a Fully Connected (FC) layer:
1. **Input**: (Batch, Window, 5) - Mixed signal + 4D One-hot control.
2. **LSTM**: 64 Hidden Units.
3. **FC**: Maps hidden state to a single scalar prediction.

## Empirical Findings & Analysis

### Noise Independence Proof
We empirically demonstrate the statistical independence of the training and testing sets by utilizing different random seeds for noise realizations.
![Noise Histogram](docs/noise_histogram.png)

### Prediction Performance
The following plots illustrate the model's ability to reconstruct the target frequency from the mixed signal.
| 1Hz Extraction | 3Hz Extraction |
| :---: | :---: |
| ![1Hz](docs/prediction_1Hz.png) | ![3Hz](docs/prediction_3Hz.png) |
| 5Hz Extraction | 7Hz Extraction |
| ![5Hz](docs/prediction_5Hz.png) | ![7Hz](docs/prediction_7Hz.png) |

### Quantitative Results (MSE)
Statistical aggregation over 3 seeds yields the following Mean Squared Error (MSE) results:

| Configuration | Mean MSE | Std Dev |
| :--- | :--- | :--- |
| **L=1** (Context Reset) | 0.609286 | 0.012273 |
| **L=100** (Stateful) | 0.556277 | 0.012773 |
| **Window=10** | 0.603791 | 0.007804 |

### Window Size Ablation
Comparing window sizes of 10 vs 100 samples shows the impact of temporal context on extraction accuracy.
![Window Size Ablation](docs/window_size_ablation.png)

## Context Reset (L=1 vs L=100) Analysis
L=1 forces the LSTM to act as a parallel filter by resetting the hidden state after every context window, preventing long-term memorization. In contrast, L=100 retains the hidden state across batches (with `shuffle=False`), presenting different performance characteristics. Our results indicate that state retention (L=100) allows the model to leverage broader temporal context, slightly improving reconstruction performance.

## Targeted Ablation Study
We performed a saliency-based pruning by identifying the top-10 hidden units most activated during 1Hz extraction.
![Ablation Plot](docs/ablation_plot.png)

**Analysis**: Targeted ablation suggests that the LSTM has learned representations in which certain hidden dimensions are disproportionately important for 1Hz reconstruction, consistent with frequency-specific feature extraction. While 1Hz extraction fails post-ablation, the 7Hz extraction remains robust.

## Setup & Usage

### Environment Setup
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Execution
To run the full pipeline and generate plots:
```bash
$env:PYTHONPATH = "."
python src/main.py
```

## Honest Assessment
The high MSE observed suggests that while the LSTM learns the frequency phase, the amplitude reconstruction is significantly dampened by the noise normalization. The model tends to underfit the clean signal's peak amplitude, likely due to the uniform noise distribution which shifts the mean of the mixed signal. Further hyperparameter tuning (e.g., deeper LSTM layers) might improve the regression accuracy.

## Dataset
Synthetic dataset generated using NumPy with reproducible random seeds (Seed 42 for Train, 123 for Test).
