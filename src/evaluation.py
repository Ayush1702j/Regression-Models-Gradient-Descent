"""Evaluation helpers."""

import pandas as pd
from .models import regression_metrics


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    metrics = regression_metrics(y_test, pred_test)
    metrics.update({"Model": name})
    return model, pred_train, pred_test, metrics


def save_results(records, path):
    df = pd.DataFrame(records)
    cols = ["Model", "MAE", "MSE", "RMSE", "R2"]
    df = df[cols]
    df.to_csv(path, index=False)
    return df
