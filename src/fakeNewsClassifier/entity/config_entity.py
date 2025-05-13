from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    fake_news_path: Path
    true_news_path: Path
    processed_x_train: Path
    processed_y_train: Path
    processed_x_test: Path
    processed_y_test: Path
    tfidf_vectoriser: Path
    test_size: float
    random_state: int
    stratify: bool


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_C: float
    params_penalty: str
    params_solver: str
    params_class_weight: str
    params_max_iter: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    base_model_path: Path
    trained_model_path: Path
    processed_x_train: Path
    processed_y_train: Path
    processed_x_test: Path
    processed_y_test: Path
    tfidf_vectoriser: Path
