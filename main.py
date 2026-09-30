"""Main experiment runner for regression models and Gradient Descent."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.data_preprocessing import prepare_data
from src.models import build_models, LinearRegressionGD, regression_metrics
from src.evaluation import evaluate_model, save_results
from src.visualization import (
    save_model_comparison,
    save_actual_vs_predicted,
    save_cost_curve,
    save_learning_rate_comparison,
    save_convergence_comparison,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
GRAPHS_DIR = ROOT / "graphs"
RESULTS_DIR.mkdir(exist_ok=True)
GRAPHS_DIR.mkdir(exist_ok=True)


def run_regression_models():
    df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data()

    records = []
    predictions = {}

    # Scale features for fair numerical conditioning across models.
    for name, model in build_models().items():
        fitted, _, pred_test, metrics = evaluate_model(
            name, model, X_train_scaled, X_test_scaled, y_train, y_test
        )
        records.append(metrics)
        predictions[name] = pred_test

    results_df = save_results(records, RESULTS_DIR / "model_results.csv")

    # Select the model with the highest R² only as an experimental reference,
    # not as a universal claim about model quality.
    best_idx = results_df["R2"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]
    save_actual_vs_predicted(y_test, predictions[best_name], GRAPHS_DIR, best_name)
    save_model_comparison(results_df, GRAPHS_DIR)

    return results_df, y_test, predictions


def run_gradient_descent(X_train_scaled, X_test_scaled, y_train, y_test):
    # Standardize target for stable gradient updates; convert predictions back.
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_train_scaled = (y_train.to_numpy() - y_mean) / y_std

    learning_rates = [0.001, 0.01, 0.05, 0.1]
    records = []
    histories = {}
    models = {}

    for lr in learning_rates:
        gd = LinearRegressionGD(learning_rate=lr, n_iterations=3000)
        gd.fit(X_train_scaled, y_train_scaled)
        pred_scaled = gd.predict(X_test_scaled)
        pred = pred_scaled * y_std + y_mean
        metrics = regression_metrics(y_test, pred)
        final_cost = gd.cost_history[-1]
        records.append({
            "Learning Rate": lr,
            "Iterations": gd.n_iterations,
            "Final Cost": final_cost,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        })
        histories[lr] = gd.cost_history
        models[lr] = (gd, pred)

    gd_df = pd.DataFrame(records)
    gd_df.to_csv(RESULTS_DIR / "gradient_descent_results.csv", index=False)

    # Use 0.05 as the main demonstration curve; the comparison table contains all rates.
    save_cost_curve(histories[0.05], GRAPHS_DIR)
    save_learning_rate_comparison(records, GRAPHS_DIR)
    save_convergence_comparison(records, GRAPHS_DIR)

    return gd_df, histories, models


def main():
    print("=" * 72)
    print("REGRESSION MODELS + GRADIENT DESCENT EXPERIMENT")
    print("Real-world application: Disease Progression Prediction")
    print("=" * 72)

    df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = prepare_data()
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print()

    results_df, y_test, predictions = run_regression_models()
    print("REGRESSION MODEL RESULTS")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    gd_df, histories, models = run_gradient_descent(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    print("GRADIENT DESCENT RESULTS")
    print(gd_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    print("Generated files:")
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        print(f"  {p.relative_to(ROOT)}")
    for p in sorted(GRAPHS_DIR.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
