from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    model_path: Path
    scaler_path: Path
    feature_columns: list
    target_column: str
    test_size: float
    random_state: int

def load_config(config_path: str = "configs/config.yaml") -> ModelConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return ModelConfig(**config) 