import logging
import sys
from pathlib import Path

import click
import pandas as pd

from pipeline.preprocessing import DataPreprocessor, PreprocessConfig
from pipeline.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """ML Pipeline — train, evaluate, and predict with sklearn models."""
    pass


@cli.command()
@click.argument("data_path")
@click.option("--target", default="target", help="Target column name")
@click.option("--model", default="random_forest",
              type=click.Choice(["logistic_regression", "random_forest", "gradient_boosting"]))
@click.option("--output-dir", default="models")
@click.option("--drop-columns", default="", help="Comma-separated columns to drop")
@click.option("--cross-validate", is_flag=True, default=False)
def train(data_path, target, model, output_dir, drop_columns, cross_validate):
    """Train a model on CSV data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset shape: {df.shape}")

    drop_cols = [c.strip() for c in drop_columns.split(",") if c.strip()]
    config = PreprocessConfig(target_column=target, drop_columns=drop_cols)
    preprocessor = DataPreprocessor(config)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform(df)

    trainer = ModelTrainer(model_name=model, output_dir=output_dir)

    if cross_validate:
        import numpy as np
        X_full = np.concatenate([X_train, X_val, X_test])
        y_full = np.concatenate([y_train, y_val, y_test])
        trainer.cross_validate(X_full, y_full)

    trainer.train(X_train, y_train, X_val, y_val)
    metrics = trainer.evaluate(X_test, y_test)
    trainer.save()

    click.echo("\n=== Results ===")
    for k, v in metrics.items():
        click.echo(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


@cli.command()
@click.argument("model_path")
@click.argument("data_path")
@click.option("--output", default="predictions.csv")
def predict(model_path, data_path, output):
    """Run inference on new data."""
    df = pd.read_csv(data_path)
    trainer = ModelTrainer.load(model_path)
    # Note: preprocessor must be loaded separately in production
    preds = trainer.predict(df.values)
    df["prediction"] = preds
    df.to_csv(output, index=False)
    click.echo(f"Predictions saved to {output}")


if __name__ == "__main__":
    cli()
