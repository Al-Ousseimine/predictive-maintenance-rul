"""Module de prétraitement et de feature engineering pour C-MAPSS."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SETTINGS_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]


@dataclass
class CMAPSSFeaturePipeline(BaseEstimator, TransformerMixin):
  """Pipeline de normalisation par régime de vol et rolling features."""

  n_regimes: int = 6
  window_size: int = 15
  rul_clip: int = 125

  scaler_settings: StandardScaler = field(default_factory=StandardScaler)
  kmeans: KMeans = field(init=False)
  regime_stats: pd.DataFrame = field(init=False)
  feature_names_: list[str] = field(default_factory=list)

  def __post_init__(self):
    self.kmeans = KMeans(
        n_clusters=self.n_regimes, random_state=42, n_init=10
    )

  def fit(self, df: pd.DataFrame, y=None):
    """Apprend les clusters opérationnels et les statistiques de chaque capteur par régime."""
    # 1. Clustering des conditions de vol
    scaled_settings = self.scaler_settings.fit_transform(df[SETTINGS_COLS])
    regimes = self.kmeans.fit_predict(scaled_settings)

    df_tmp = df.copy()
    df_tmp["op_regime"] = regimes

    # 2. Statistiques par régime pour chaque capteur (Train uniquement)
    self.regime_stats = (
        df_tmp.groupby("op_regime")[SENSOR_COLS]
        .agg(["mean", "std"])
        .reset_index()
    )
    return self

  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    """Applique la normalisation conditionnelle et génère les fenêtres glissantes."""
    df_out = df.copy()

    # 1. Assignation des régimes
    scaled_settings = self.scaler_settings.transform(df_out[SETTINGS_COLS])
    df_out["op_regime"] = self.kmeans.predict(scaled_settings)

    # 2. Normalisation par régime (z-score local)
    for s in SENSOR_COLS:
      mean_map = self.regime_stats.set_index("op_regime")[(s, "mean")].to_dict()
      std_map = self.regime_stats.set_index("op_regime")[(s, "std")].to_dict()

      r_mean = df_out["op_regime"].map(mean_map)
      r_std = df_out["op_regime"].map(std_map).replace(0, 1.0)
      df_out[f"{s}_norm"] = (df_out[s] - r_mean) / r_std

    norm_sensors = [f"{s}_norm" for s in SENSOR_COLS]

    # 3. Rolling features (par moteur pour respecter les frontières temporelles)
    grouped = df_out.groupby("unit_id")[norm_sensors]

    roll_means = (
        grouped.rolling(window=self.window_size, min_periods=1).mean().values
    )
    roll_stds = (
        grouped.rolling(window=self.window_size, min_periods=1).std().values
    )

    mean_cols = [f"{c}_roll_mean_{self.window_size}" for c in norm_sensors]
    std_cols = [f"{c}_roll_std_{self.window_size}" for c in norm_sensors]

    df_out[mean_cols] = roll_means
    df_out[std_cols] = np.nan_to_num(roll_stds, nan=0.0)

    # 4. Clipping de la RUL si présente
    if "rul" in df_out.columns:
      df_out["rul_clipped"] = df_out["rul"].clip(upper=self.rul_clip)

    self.feature_names_ = norm_sensors + mean_cols + std_cols
    return df_out