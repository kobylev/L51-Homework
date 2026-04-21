from dataclasses import dataclass, field
from pathlib import Path
import torch
from typing import List

@dataclass
class Config:
    # Signal Parameters
    FREQUENCIES: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    SAMPLE_RATE: int = 1000
    DURATION: int = 10
    NUM_SAMPLES: int = 10000 # SAMPLE_RATE * DURATION
    
    # Data Parameters
    CONTEXT_WINDOW: int = 100
    TRAIN_SPLIT: float = 0.8
    
    # Model Parameters
    INPUT_SIZE: int = 5 # 1 (mixed signal) + 4 (one-hot control)
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 1
    OUTPUT_SIZE: int = 1
    
    # Training Parameters
    BATCH_SIZE: int = 64
    EPOCHS: int = 3
    LEARNING_RATE: float = 0.001
    L_PARAMETER: int = 1
    NUM_SEEDS: int = 3
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "output" / "analysis"
    SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"
    MODEL_L1_PATH: Path = BASE_DIR / "output" / "model_l1.pt"
    MODEL_L100_PATH: Path = BASE_DIR / "output" / "model_l100.pt"

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
