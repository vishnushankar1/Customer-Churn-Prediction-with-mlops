import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from src.logger import logging
from src.exception import CustomException
from src.utils.common import read_yaml_file
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact
)


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            schema_path = os.path.join("config", "schema.yaml")
            self._schema_config = read_yaml_file(schema_path)
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self):
        try:
            logging.info("🚀 Starting Data Transformation")

            # Load validated datasets
            train_df = pd.read_csv(self.data_validation_artifact.validated_train_path)
            test_df = pd.read_csv(self.data_validation_artifact.validated_test_path)

            # Clean column names (remove leading/trailing spaces)
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Target Column
            target_col = "Churn"  # since not defined in schema
            num_cols = self._schema_config["numerical_columns"]
            cat_cols = self._schema_config["categorical_columns"]

            # Encode Target Column
            train_df[target_col] = train_df[target_col].map({"No": 0, "Yes": 1})
            test_df[target_col] = test_df[target_col].map({"No": 0, "Yes": 1})

            # One-Hot Encode Categorical Columns (excluding target)
            cat_cols = [col for col in cat_cols if col != target_col]

            train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
            test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)


            # Align Columns
            train_df, test_df = train_df.align(test_df, join="outer", axis=1, fill_value=0)
            # Convert all boolean or object columns to int (to ensure no True/False or strings remain)
            train_df = train_df.apply(lambda col: col.astype(int) if col.dtypes == bool else col)
            test_df = test_df.apply(lambda col: col.astype(int) if col.dtypes == bool else col)

            # Split target
            target_col = self._schema_config.get("target_column", "Churn")

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

           
            transform_dir = os.path.join("artifacts", "transformation")
            os.makedirs(transform_dir, exist_ok=True)

            # ✅ File paths inside transformation folder
            X_train_path = os.path.join(transform_dir, "X_train.npy")
            y_train_path = os.path.join(transform_dir, "y_train.npy")
            X_test_path = os.path.join(transform_dir, "X_test.npy")
            y_test_path = os.path.join(transform_dir, "y_test.npy")
            scaler_path = os.path.join(transform_dir, "scaler.pkl")

            # ✅ Save NumPy arrays
            np.save(X_train_path, X_train_scaled)
            np.save(y_train_path, y_train)
            np.save(X_test_path, X_test_scaled)
            np.save(y_test_path, y_test)
            
            # Save scaler
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            logging.info("✅ Data transformation complete")

            return DataTransformationArtifact(
                preprocessor_object_path=scaler_path,
                X_train_path=X_train_path,
                y_train_path=y_train_path,
                X_test_path=X_test_path,
                y_test_path=y_test_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_validation import DataValidation

        ingestion = DataIngestion()
        ingestion_artifact = ingestion.initiate_data_ingestion()

        validator = DataValidation(data_ingestion_artifact=ingestion_artifact)
        validation_artifact = validator.initiate_data_validation()

        transformer = DataTransformation(
            data_ingestion_artifact=ingestion_artifact,
            data_validation_artifact=validation_artifact
        )

        transformation_artifact = transformer.transform()

        print("✅ Transformation Complete!")
        print(f"📁 X_train saved at: {transformation_artifact.X_train_path}")
        print(f"📁 y_train saved at: {transformation_artifact.y_train_path}")
        print(f"📁 X_test saved at: {transformation_artifact.X_test_path}")
        print(f"📁 y_test saved at: {transformation_artifact.y_test_path}")
        print(f"⚙️  Scaler saved at: {transformation_artifact.preprocessor_object_path}")

    except Exception as e:
        raise CustomException(e, sys)
