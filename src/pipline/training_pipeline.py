import sys
import os
sys.path.append(os.getcwd())

from src.components.data_ingestion import DataIngestion
from src.entity.artifact_entity import DataIngestionArtifact

def run_pipeline():
    ingestion = DataIngestion()
    artifact = ingestion.initiate_data_ingestion()
    print("✅ Data ingestion completed!")
    print(f"Train file: {artifact.train_file_path}")
    print(f"Test file: {artifact.test_file_path}")

if __name__ == "__main__":
    run_pipeline()
