# src/entity/artifact_entity.py

from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str
    validated_train_path: str
    validated_test_path: str

@dataclass
class DataTransformationArtifact:
    preprocessor_object_path: str
    X_train_path: str
    y_train_path: str
    X_test_path: str
    y_test_path: str

@dataclass
class ModelTrainerArtifact:
    model_path: str
    metrics_path: str
