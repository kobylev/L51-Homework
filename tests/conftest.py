import pytest
import torch
import numpy as np
from config.config import Config
from src.models.model import ConditionalLSTM
from src.data.dataset import generate_signal, BandpassDataset

@pytest.fixture
def config():
    return Config(EPOCHS=1, NUM_SEEDS=1)

@pytest.fixture
def sample_batch(config):
    mixed, clean, _ = generate_signal(config.FREQUENCIES, config.SAMPLE_RATE, 1)
    dataset = BandpassDataset(mixed, clean, config.CONTEXT_WINDOW)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE)
    return next(iter(loader))

@pytest.fixture
def model(config):
    return ConditionalLSTM(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.OUTPUT_SIZE)
