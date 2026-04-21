# TODO: Conditional LSTM Bandpass Filter

## Current State: [COMPLETED]
The project has a fully functional implementation with the following components:

- [x] **Modular Structure:** Code split into `config.py`, `datasets.py`, `model.py`, `train.py`, `evaluate.py`, and `main.py`.
- [x] **Signal Generation:** Synthetic signal mixture with random amplitude/phase noise and independent realizations for Train/Test.
- [x] **Model Implementation:** Conditional LSTM that takes mixed signal + one-hot control vector.
- [x] **Statistical Aggregation:** Main script runs over multiple seeds and reports Mean +/- STD for MSE.
- [x] **Ablation Studies:**
    - [x] $L=1$ vs $L=100$ (Hidden state carry-over).
    - [x] Window size ablation ($W=10$ vs $W=100$).
    - [x] Targeted unit ablation (Zeroing top hidden units for specific frequencies).
- [x] **Visualization Suite:**
    - [x] Time-domain prediction plots for all frequencies.
    - [x] Noise distribution histograms.
    - [x] Window size ablation bar charts.
    - [x] Targeted ablation impact plots.

---

## Future Improvements

### 1. Model Optimization & Deepening
- [ ] **Hyperparameter Tuning:** Use Optuna or Ray Tune to optimize `HIDDEN_SIZE`, `LEARNING_RATE`, and `NUM_LAYERS`.
- [ ] **Deep Architectures:** Experiment with multi-layer LSTMs or GRUs to improve filtering precision.
- [ ] **Attention Mechanism:** Add a temporal attention layer to allow the model to focus on specific cycles of the signal.

### 2. Signal Complexity
- [ ] **Real-world Noise:** Test the model against non-Gaussian noise, white noise, and environmental recordings.
- [ ] **Dynamic Frequencies:** Allow target frequencies to shift over time (Chirp signals) and test if the LSTM can track them.
- [ ] **Overlap/Interference:** Increase the number of frequencies or use frequencies that are much closer together (e.g., 5Hz and 5.5Hz).

### 3. Training & Evaluation Enhancements
- [ ] **Early Stopping:** Implement early stopping based on validation loss to prevent overfitting in long training runs.
- [ ] **Cross-Frequency Interference Matrix:** Create a heatmap showing how much each frequency's model-sub-network interferes with others.
- [ ] **Deployment Preparation:** Export the model to TorchScript or ONNX for real-time inference in C++ or Edge devices.

### 4. Documentation & Reproducibility
- [ ] **Interactive Notebook:** Create a Jupyter Notebook demonstrating the filtering in real-time with an interactive slider for the control vector.
- [ ] **CI/CD Integration:** Add GitHub Actions to run the statistical test suite on every push.
