# PRD: Conditional LSTM Bandpass Filter

## 1. Product Vision & Goals
### Vision
To develop a deep learning-based signal processing tool capable of dynamically extracting specific frequency components from a noisy mixture using a conditional LSTM architecture. Unlike static filters, this model learns to isolate frequencies based on a real-time control signal.

### Goals
*   **Dynamic Filtering:** Isolate specific frequencies (1Hz, 3Hz, 5Hz, 7Hz) from a composite signal.
*   **Robustness:** Demonstrate filtering capability under variable amplitude (0.8–1.2) and phase (0–2π) noise.
*   **Interpretability:** Prove that specific LSTM hidden units specialize in different frequency extraction tasks via targeted ablation.
*   **Reliability:** Validate performance through statistical aggregation over multiple random seeds and ensure noise independence between training and testing sets.

## 2. Functional Requirements
### 2.1 Signal Synthesis
*   **Composite Signal:** Generate a mixture of four sine waves ($f \in \{1, 3, 5, 7\}$ Hz).
*   **Noise Injection:** Apply independent random amplitude and phase shifts to each component in every realization.
*   **Control Vector:** Provide a one-hot vector $[C_1, C_3, C_5, C_7]$ to specify the target frequency.
*   **Train/Test Split:** Implement an 80/20 split using entirely independent noise realizations for each set.

### 2.2 Model Architecture
*   **Input Layer:** 5 dimensions (1 mixed signal value + 4-bit one-hot control).
*   **Recurrent Layer:** LSTM layer to capture temporal dependencies.
*   **Output Layer:** Single linear unit predicting the instantaneous value of the clean target signal.
*   **Conditional Logic:** The model must use the control vector at every time step to modulate its internal state/output.

### 2.3 Evaluation & Ablation
*   **MSE Benchmarking:** Calculate Mean Squared Error between predicted and clean target signals.
*   **State Carry-over Test ($L$ parameter):** Compare performance when hidden states are reset every window ($L=1$) versus preserved across windows ($L=100$).
*   **Window Size Ablation:** Compare $W=10$ vs. $W=100$ to determine the minimum temporal context required for effective filtering.
*   **Targeted Unit Ablation:** Identify top-K hidden units for a specific frequency (e.g., 1Hz) and zero them out to observe the impact on that frequency versus others.

### 2.4 Visualization
*   **Time-domain Plots:** Overlays of ground truth vs. predicted signals for each frequency.
*   **Noise Distribution:** Histograms of amplitude and phase noise for Train and Test sets to prove independence.
*   **Ablation Plots:** Bar charts for window size comparison and signal degradation plots for targeted unit ablation.

## 3. Technical Architecture
### 3.1 Stack
*   **Framework:** PyTorch (Model definition, Training, Evaluation).
*   **Data Handling:** NumPy for signal generation, PyTorch `Dataset` and `DataLoader` for batching.
*   **Visualization:** Matplotlib for generating all required plots.

### 3.2 Modular Design
*   `config.py`: Centralized hyperparameters and constants.
*   `datasets.py`: Signal generation and data augmentation logic.
*   `model.py`: `ConditionalLSTM` class definition.
*   `train.py`: Standardized training and evaluation loops.
*   `evaluate.py`: Specialized functions for ablation studies and visualization.
*   `main.py`: Orchestration of the full experimental suite.

## 4. Success Metrics
*   **Primary Metric:** Mean Squared Error (MSE) < 0.05 on the test set for all frequencies.
*   **Statistical Proof:** Standard deviation across 3+ seeds should be within 10% of the mean MSE.
*   **Generalization:** The model must maintain low MSE on test data with noise distributions statistically identical but numerically distinct from training data.
*   **Selectivity:** Targeted ablation of 1Hz units should significantly degrade 1Hz performance while leaving 7Hz performance relatively intact.
