# src/components/data_ingestion.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import DataIngestionArtifact

# Ensure the project root is in the path (for DVC and CLI)
sys.path.append(os.getcwd())


class DataIngestion:
    def __init__(self,
                 raw_data_path="data/data.csv",
                 train_path="artifacts/data_ingestion/train.csv",
                 test_path="artifacts/data_ingestion/test.csv",
                 test_size=0.20):
        self.raw_data_path = raw_data_path
        self.train_path = train_path
        self.test_path = test_path
        self.test_size = test_size

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("📥 Starting data ingestion process.")

        try:
            df = pd.read_csv(self.raw_data_path)
            logging.info(f"✅ Dataset loaded successfully with shape: {df.shape}")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.train_path), exist_ok=True)

            # Split data
            train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=42)

            # Save split files
            train_df.to_csv(self.train_path, index=False)
            test_df.to_csv(self.test_path, index=False)

            logging.info("✅ Train and test files created successfully.")
            logging.info(f"Train path: {self.train_path}")
            logging.info(f"Test path: {self.test_path}")

            return DataIngestionArtifact(
                train_file_path=self.train_path,
                test_file_path=self.test_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    artifact = ingestion.initiate_data_ingestion()
    print("✅ Data ingestion completed.")
    print(f"Train file saved at: {artifact.train_file_path}")
    print(f"Test file saved at: {artifact.test_file_path}")
