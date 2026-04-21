import pytest
import torch
from src.evaluation.evaluator import run_evaluation
from src.training.trainer import train_one_seed

def test_data_shapes(sample_batch, config):
    x, y = sample_batch
    # x: (batch, window, 1+4)
    assert x.shape == (config.BATCH_SIZE, config.CONTEXT_WINDOW, config.INPUT_SIZE)
    # y: (batch, window, 1)
    assert y.shape == (config.BATCH_SIZE, config.CONTEXT_WINDOW, 1)

def test_model_forward(model, sample_batch, config):
    x, _ = sample_batch
    output, _ = model(x)
    assert output.shape == (config.BATCH_SIZE, config.CONTEXT_WINDOW, 1)

def test_performance_threshold(config):
    # Short training to check if MSE is somewhat reasonable
    model = train_one_seed(config, seed=0)
    metrics = run_evaluation(model, config)
    # Since it's a normalized signal (divided by 4), MSE should be < 1.0 even with minimal training
    assert metrics['mse'] < 1.0
