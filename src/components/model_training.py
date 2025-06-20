import os
import sys
import json
import numpy as np
import mlflow
import mlflow.keras
from dataclasses import dataclass
from dotenv import load_dotenv
from dagshub import dagshub_logger, init
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerArtifact:
    model_path: str
    metrics_path: str


class ModelTrainer:
    def __init__(self):
        try:
            # Load credentials from .env
            load_dotenv()
            username = os.getenv("MLFLOW_TRACKING_USERNAME")
            token = os.getenv("MLFLOW_TRACKING_PASSWORD")

            # DagsHub MLflow setup
            mlflow.set_tracking_uri("https://dagshub.com/vishnushankar1/Customer-Churn-Prediction-with-mlops.mlflow")
            mlflow.set_experiment("Customer-Churn-ANN")

            # Setup local artifact dir (for model.h5 & metrics.json)
            self.artifact_dir = os.path.join("artifacts", "model_training")
            os.makedirs(self.artifact_dir, exist_ok=True)

            # DagsHub integration
            init(repo_owner="vishnushankar1", repo_name="Customer-Churn-Prediction-with-mlops", mlflow=True)

        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):
        try:
            X_train = np.load("artifacts/transformation/X_train.npy")
            y_train = np.load("artifacts/transformation/y_train.npy")
            X_test = np.load("artifacts/transformation/X_test.npy")
            y_test = np.load("artifacts/transformation/y_test.npy")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def build_model(self, input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def evaluate(self, model, X_test, y_test):
        y_pred_prob = model.predict(X_test).ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_prob)
        }

    def train(self) -> ModelTrainerArtifact:
        try:
            logging.info("🏋️ Starting model training...")
            X_train, y_train, X_test, y_test = self.load_data()
            model = self.build_model(input_dim=X_train.shape[1])

            early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

            with mlflow.start_run():
                history = model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=1
                )

                metrics = self.evaluate(model, X_test, y_test)

                # Log metrics to MLflow
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Save model locally and log to DagsHub
                model_path = os.path.join(self.artifact_dir, "model.h5")
                model.save(model_path)
                mlflow.log_artifact(model_path, artifact_path="model")


                # Save metrics as JSON
                metrics_path = os.path.join(self.artifact_dir, "metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)

                mlflow.log_artifact(model_path)
                mlflow.log_artifact(metrics_path)

                logging.info("✅ Model training and MLflow logging completed.")

                return ModelTrainerArtifact(
                    model_path=model_path,
                    metrics_path=metrics_path
                )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        artifact = trainer.train()
        print("✅ Model Training Complete!")
        print(f"📦 Model saved at: {artifact.model_path}")
        print(f"📊 Metrics saved at: {artifact.metrics_path}")
        print("🔗 View your MLflow logs on DagsHub: https://dagshub.com/vishnushankar1/Customer-Churn-Prediction-with-mlops.mlflow")
    except Exception as e:
        print(CustomException(e, sys))
