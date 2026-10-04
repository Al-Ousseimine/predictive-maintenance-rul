"""Script d'entraînement et d'évaluation automatisé pour C-MAPSS."""

import argparse
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from src.data.loader import compute_train_rul, load_cmapss_subset
from src.features.engineering import CMAPSSFeaturePipeline


def compute_cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Métrique officielle NASA C-MAPSS."""
  d = np.array(y_pred) - np.array(y_true)
  score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
  return float(np.sum(score))


def run_training(
    data_dir: str = "data/raw",
    output_dir: str = "models",
    subset: str = "FD004",
    n_splits: int = 5,
    rul_clip: int = 125,
):
  raw_path = Path(data_dir)
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  print(f"[1/4] Chargement des données brutes ({subset})...")
  df_train, df_test, df_rul = load_cmapss_subset(raw_path, subset=subset)
  df_train = compute_train_rul(df_train)

  print("[2/4] Transformation des données (Feature Engineering)...")
  pipeline = CMAPSSFeaturePipeline(rul_clip=rul_clip)
  df_train_proc = pipeline.fit_transform(df_train)
  df_test_proc = pipeline.transform(df_test)

  # Sauvegarde du pipeline de transformation
  joblib.dump(pipeline, out_path / f"pipeline_{subset}.joblib")

  feature_cols = pipeline.feature_names_
  X_train = df_train_proc[feature_cols]
  y_train = df_train_proc["rul_clipped"]
  groups = df_train_proc["unit_id"]

  # Extraction du dernier cycle pour les moteurs de test
  X_test = df_test_proc.groupby("unit_id")[feature_cols].last()
  y_test = df_rul["rul"].clip(upper=rul_clip).values

  print(
      f"[3/4] Entraînement LightGBM avec GroupKFold ({n_splits} splits sur"
      " unit_id)..."
  )
  gkf = GroupKFold(n_splits=n_splits)

  oof_preds = np.zeros(len(X_train))
  test_preds = np.zeros(len(X_test))
  trained_models = []

  lgb_params = {
      "objective": "regression",
      "metric": "rmse",
      "boosting_type": "gbdt",
      "learning_rate": 0.05,
      "num_leaves": 31,
      "max_depth": 6,
      "feature_fraction": 0.8,
      "bagging_fraction": 0.8,
      "bagging_freq": 1,
      "reg_alpha": 1.0,
      "reg_lambda": 5.0,
      "random_state": 42,
      "verbose": -1,
      "n_jobs": -1,
  }

  for fold, (t_idx, v_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    X_tr, y_tr = X_train.iloc[t_idx], y_train.iloc[t_idx]
    X_va, y_va = X_train.iloc[v_idx], y_train.iloc[v_idx]

    model = lgb.LGBMRegressor(**lgb_params, n_estimators=600)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    oof_preds[v_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / n_splits
    trained_models.append(model)

  # Sauvegarde des modèles
  joblib.dump(trained_models, out_path / f"lgb_models_{subset}.joblib")

  # Métriques finales
  cv_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
  test_preds_clipped = np.clip(test_preds, a_min=0, a_max=rul_clip)

  test_rmse = np.sqrt(mean_squared_error(y_test, test_preds_clipped))
  test_mae = mean_absolute_error(y_test, test_preds_clipped)
  test_score = compute_cmapss_score(y_test, test_preds_clipped)

  print("\n" + "=" * 45)
  print(f"RÉSULTATS DE VALIDATION ({subset})")
  print("=" * 45)
  print(f"RMSE Out-Of-Fold (CV) : {cv_rmse:.2f} cycles")
  print(f"RMSE Test             : {test_rmse:.2f} cycles")
  print(f"MAE Test              : {test_mae:.2f} cycles")
  print(f"Score NASA Test       : {test_score:.2f}")
  print("=" * 45)
  print(f"Modèles et pipeline sauvegardés dans : {out_path.resolve()}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Pipeline d'entraînement C-MAPSS"
  )
  parser.add_argument("--data-dir", type=str, default="data/raw")
  parser.add_argument("--output-dir", type=str, default="models")
  parser.add_argument("--subset", type=str, default="FD004")
  parser.add_argument("--n-splits", type=int, default=5)
  parser.add_argument("--rul-clip", type=int, default=125)

  args = parser.parse_args()
  run_training(
      data_dir=args.data_dir,
      output_dir=args.output_dir,
      subset=args.subset,
      n_splits=args.n_splits,
      rul_clip=args.rul_clip,
  )