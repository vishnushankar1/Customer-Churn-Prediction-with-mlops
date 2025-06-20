import json
import sys
import os
import pandas as pd
from pandas import DataFrame

from src.exception import CustomException
from src.logger import logging
from src.utils.common import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact

            schema_path = os.path.join("config", "schema.yaml")
            self._schema_config = read_yaml_file(file_path=schema_path)

            # ✅ Updated paths to use data_validation folder
            self.data_validation_dir = os.path.join("artifacts", "data_validation")
            os.makedirs(self.data_validation_dir, exist_ok=True)

            self.validated_train_path = os.path.join(self.data_validation_dir, "validated_train.csv")
            self.validated_test_path = os.path.join(self.data_validation_dir, "validated_test.csv")
            self.validation_report_path = os.path.join(self.data_validation_dir, "validation_report.json")

        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column count matching: [{status}]")
            return status
        except Exception as e:
            raise CustomException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("🧪 Starting data validation process.")

            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            for df, name in [(train_df, "train"), (test_df, "test")]:
                if "customerID" in df.columns:
                    df.drop("customerID", axis=1, inplace=True)
                    logging.info(f"Dropped customerID from {name} data.")

                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                before = len(df)
                df.dropna(inplace=True)
                logging.info(f"Dropped {before - len(df)} NaN rows from {name} data.")

            # ✅ Save validated CSVs
            train_df.to_csv(self.validated_train_path, index=False)
            test_df.to_csv(self.validated_test_path, index=False)

            if not self.validate_number_of_columns(train_df):
                validation_error_msg += "Column count mismatch in train data. "
            if not self.validate_number_of_columns(test_df):
                validation_error_msg += "Column count mismatch in test data. "

            if not self.is_column_exist(train_df):
                validation_error_msg += "Missing required columns in train data. "
            if not self.is_column_exist(test_df):
                validation_error_msg += "Missing required columns in test data. "

            validation_status = len(validation_error_msg) == 0

            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(self.validation_report_path, "w") as f:
                json.dump(validation_report, f, indent=4)

            logging.info(f"✅ Validation completed. Status: {validation_status}")

            return DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg.strip(),
                validation_report_file_path=self.validation_report_path,
                validated_train_path=self.validated_train_path,
                validated_test_path=self.validated_test_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion

        ingestion = DataIngestion()
        ingestion_artifact = ingestion.initiate_data_ingestion()

        validator = DataValidation(data_ingestion_artifact=ingestion_artifact)
        validation_artifact = validator.initiate_data_validation()

        print("✅ Validation Done")
        print(f"Report Path: {validation_artifact.validation_report_file_path}")

    except Exception as e:
        print(CustomException(e, sys))
