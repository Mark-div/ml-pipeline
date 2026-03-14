import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


class ModelTrainer:
    def __init__(self, model_name: str = "random_forest", output_dir: str = "models"):
        assert model_name in MODELS, f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}"
        self.model_name = model_name
        self.model = MODELS[model_name]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        logger.info(f"Training {self.model_name} on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        logger.info("Training complete")

        train_preds = self.model.predict(X_train)
        self.metrics["train_accuracy"] = accuracy_score(y_train, train_preds)
        logger.info(f"Train accuracy: {self.metrics['train_accuracy']:.4f}")

        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            self.metrics["val_accuracy"] = accuracy_score(y_val, val_preds)
            self.metrics["val_f1"] = f1_score(y_val, val_preds, average="weighted")
            logger.info(f"Val accuracy: {self.metrics['val_accuracy']:.4f} | F1: {self.metrics['val_f1']:.4f}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        preds = self.model.predict(X_test)
        proba = self.model.predict_proba(X_test) if hasattr(self.model, "predict_proba") else None

        self.metrics.update({
            "test_accuracy": accuracy_score(y_test, preds),
            "test_precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "test_recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "test_f1": f1_score(y_test, preds, average="weighted", zero_division=0),
        })

        if proba is not None and len(np.unique(y_test)) == 2:
            self.metrics["test_roc_auc"] = roc_auc_score(y_test, proba[:, 1])

        logger.info(f"Test accuracy: {self.metrics['test_accuracy']:.4f}")
        logger.info(f"Test F1: {self.metrics['test_f1']:.4f}")
        logger.info("\n" + classification_report(y_test, preds))

        return self.metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        logger.info(f"Running {cv}-fold cross validation...")
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        result = {"cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()}
        logger.info(f"CV F1: {result['cv_f1_mean']:.4f} ± {result['cv_f1_std']:.4f}")
        return result

    def save(self, name: Optional[str] = None) -> str:
        name = name or self.model_name
        model_path = self.output_dir / f"{name}.pkl"
        metrics_path = self.output_dir / f"{name}_metrics.json"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    @classmethod
    def load(cls, model_path: str) -> "ModelTrainer":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        trainer = cls.__new__(cls)
        trainer.model = model
        trainer.model_name = Path(model_path).stem
        trainer.metrics = {}
        return trainer

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError(f"{self.model_name} does not support probability estimates")
