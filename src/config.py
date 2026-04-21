import torch

class Config:
    # Signal Parameters
    FREQUENCIES = [1, 3, 5, 7]
    SAMPLE_RATE = 1000
    DURATION = 10
    NUM_SAMPLES = SAMPLE_RATE * DURATION
    
    # Data Parameters
    CONTEXT_WINDOW = 100 # Default, will ablate with 10
    TRAIN_SPLIT = 0.8
    
    # Model Parameters
    INPUT_SIZE = 5 # 1 (mixed signal) + 4 (one-hot control)
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    OUTPUT_SIZE = 1
    
    # Training Parameters
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 0.001
    L_PARAMETER = 1 # Reset hidden state every window
    NUM_SEEDS = 3
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    DOCS_DIR = "docs"
    PLOTS_DIR = "docs/plots"
    OUTPUTS_DIR = "outputs"
    MODEL_L1_PATH = "outputs/model_l1.pt"
    MODEL_L100_PATH = "outputs/model_l100.pt"
