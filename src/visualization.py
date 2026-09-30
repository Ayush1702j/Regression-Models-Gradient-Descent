"""Plot generation for the experiment."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_model_comparison(results, out_dir):
    out_dir = Path(out_dir)
    models = results["Model"].tolist()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.25
    plt.bar(x - width, results["MAE"], width, label="MAE")
    plt.bar(x, results["RMSE"], width, label="RMSE")
    plt.bar(x + width, results["R2"] * 100, width, label="R² × 100")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Metric value")
    plt.title("Regression Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=180)
    plt.close()


def save_actual_vs_predicted(y_true, y_pred, out_dir, model_name):
    out_dir = Path(out_dir)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.75)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual target")
    plt.ylabel("Predicted target")
    plt.title(f"Actual vs Predicted — {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted.png", dpi=180)
    plt.close()


def save_cost_curve(cost_history, out_dir):
    out_dir = Path(out_dir)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cost_history) + 1), cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(θ)")
    plt.title("Gradient Descent Cost vs Iteration")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "gradient_descent_cost.png", dpi=180)
    plt.close()


def save_learning_rate_comparison(records, out_dir):
    out_dir = Path(out_dir)
    x = [r["Learning Rate"] for r in records]
    cost = [r["Final Cost"] for r in records]
    plt.figure(figsize=(8, 5))
    plt.plot(x, cost, marker="o")
    plt.xlabel("Learning rate")
    plt.ylabel("Final cost")
    plt.title("Learning Rate vs Final Gradient Descent Cost")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_rate_vs_cost.png", dpi=180)
    plt.close()


def save_convergence_comparison(records, out_dir):
    out_dir = Path(out_dir)
    x = [str(r["Learning Rate"]) for r in records]
    iterations = [r["Iterations"] for r in records]
    plt.figure(figsize=(8, 5))
    plt.bar(x, iterations)
    plt.xlabel("Learning rate")
    plt.ylabel("Iterations")
    plt.title("Gradient Descent Iterations by Learning Rate")
    plt.tight_layout()
    plt.savefig(out_dir / "learning_rate_iterations.png", dpi=180)
    plt.close()
