"""Configuration management for churn prediction system."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChurnPredictionConfig:
    """Configuration class for churn prediction system."""
    
    # Data processing
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Model architecture
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.3
    activation: str = 'relu'
    output_activation: str = 'sigmoid'
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 10
    
    # Evaluation
    classification_threshold: float = 0.5
    
    # File paths
    data_file: str = '../data/churn-bigml-80.csv'
    model_save_path: str = '../models/churn_model.h5'
    
    # Feature definitions
    categorical_features: List[str] = field(default_factory=lambda: [
        'State', 'International plan', 'Voice mail plan'
    ])