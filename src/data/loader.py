"""" Mon Module de chargement des doneés"""

from pathlib import Path
import pandas as pd

# Définition des 26 colonnes standard du dataset C-MAPSS
INDEX_COLS =  ["unit_id", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMN_NAMES = INDEX_COLS + SETTING_COLS + SENSOR_COLS

def load_cmapss_subset(data_path: str | Path, subset: str = "FD004") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """"Charge les ensembles train, test et les RUL cibles su sous-ensemble choisi:
    Args:
       data_path: Chemein vers le dossier contenant les fichiers bruts (ex: data/raw).
       subset: Identifiant du jeu (par défaut FD004).

    Returns:
    Tupe contenant (def_train, df_test, df_rul).
    """
    path = Path(data_path)

    train_file = path / f"train_{subset}.txt"
    test_file = path / f"test_{subset}.txt"
    rul_file = path / f"RUL_{subset}.txt"

    # Lecture des fichiers séparés par des espaces (ignorer les espaces blancs multiples )
    df_train = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df_test = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMN_NAMES)

    # Le fichier RUL ne contient qu'un seule valeur par moteur testé
    df_rul = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["rul"])

    return df_train, df_test, df_rul

def compute_train_rul(df_train: pd.DataFrame) -> pd.DataFrame:
    """"Calcule la RUL réelle dégressif (cycle maximal - cycle courant) pour l'entrainement."""
    df = df_train.copy()
    ## La durée de vie restante RUL (Remaining Useful Life) est le nombre de cycle qu'il reste à un moteur avant de tomber en panne. Dans le dataset, les moteurs fonctionnent jusqu'à la panne complète. Mais le fichier brut ne donne pas directement la colonne rul : il donne seulement le moteur (unit_id) et le cycle en cours (time_cycles).
    max_cycles = df.groupby("unit_id")["time_cycles"].transform("max") # trouve le dernier cycle enregistré pour chaque moteur, ie le cycle le plus élevé pour chaque moteur
    df["rul"] = max_cycles - df["time_cycles"] # on soustrait le cycle actuel de la panne au cycle max. Ce qui donne le nombre de cycle restant avant de tomber en panne
    return df
