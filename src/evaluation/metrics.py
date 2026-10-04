"""Module d'évaluation des performances pour C-MAPSS."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Calcule le score asymétrique officiel de la NASA (C-MAPSS).

  d = y_pred - y_true
  Pénalité = exp(-d / 13) - 1 si d < 0 (prédiction en avance / conservatrice)
           = exp(d / 10) - 1  si d >= 0 (prédiction en retard / critique)
  """
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)

  d = y_pred - y_true
  penalties = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
  return float(np.sum(penalties))


def evaluate_rul_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> dict[str, Any]:
  """Génère un dictionnaire complet des métriques de régression."""
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)

  rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
  mae = float(mean_absolute_error(y_true, y_pred))
  r2 = float(r2_score(y_true, y_pred))
  nasa_score = compute_cmapss_score(y_true, y_pred)

  # Calcul du ratio d'erreurs en retard (sur-estimation de la RUL = risque de panne en vol)
  d = y_pred - y_true
  late_pred_ratio = float(np.mean(d > 0) * 100.0)

  return {
      "model": model_name,
      "rmse": round(rmse, 2),
      "mae": round(mae, 2),
      "r2": round(r2, 4),
      "nasa_score": round(nasa_score, 2),
      "late_prediction_pct": round(late_pred_ratio, 1),
  }


def create_evaluation_report(
    eval_results: list[dict[str, Any]],
) -> pd.DataFrame:
  """Convertit une liste de métriques en tableau comparatif ordonné."""
  df = pd.DataFrame(eval_results)
  return df.sort_values(by="nasa_score", ascending=True).reset_index(drop=True)