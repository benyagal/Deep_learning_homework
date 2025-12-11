"""
A projekt fő belépési pontja.
Ez a szkript vezérli az adat-előfeldolgozást és a modell tanítását.
"""
import random
import numpy as np
import torch
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from mord import LogisticAT

# A projekt gyökerét hozzáadjuk a Python útvonalhoz, hogy a modulok importálhatók legyenek
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.data_preprocessing import get_processed_data
from src.train import run_training
from src.utils import get_logger

logger = get_logger(__name__)

def run_baseline_model(df, feature_cols):
    """Futtatja és kiértékeli a baseline modellt (Ordinális Logisztikus Regresszió)."""
    logger.info("--- Baseline Modell (Ordinális Regresszió) futtatása ---")
    
    X = df[feature_cols].values
    y = df['label_int'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    model = LogisticAT(alpha=0.5) # alpha=0 -> standard logisztikus regresszió
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    
    logger.info(f"Baseline Modell MAE: {mae:.4f}")
    return mae

def main():
    """A teljes folyamatot vezérlő fő függvény."""
    
    logger.info("=================================================")
    logger.info("      ÁSZF Érthetőség Predikciós Folyamat Indítása     ")
    logger.info("=================================================")

    # Reprodukálhatóság biztosítása
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    # --- Konfiguráció naplózása ---
    logger.info("--- Konfigurációs paraméterek ---")
    logger.info(f"SEED: {config.SEED}")
    logger.info(f"DEVICE: {config.DEVICE}")
    logger.info(f"MODEL_NAME: {config.MODEL_NAME}")
    logger.info(f"MAX_LEN: {config.MAX_LEN}")
    logger.info(f"BATCH_SIZE: {config.BATCH_SIZE}")
    logger.info(f"EPOCHS: {config.EPOCHS}")
    logger.info(f"LEARNING_RATE: {config.LEARNING_RATE}")
    logger.info(f"KFOLDS: {config.KFOLDS}")
    logger.info(f"PATIENCE: {config.PATIENCE}")
    logger.info("---------------------------------")

    logger.info("--- 1. Adat-előfeldolgozás ---")
    df_processed = get_processed_data(config.ANNOTATION_FILE, logger)

    if df_processed is not None and not df_processed.empty:
        # --- Baseline modell futtatása ---
        run_baseline_model(df_processed, config.FEATURE_COLS)

        # --- Transformer modell tanítása ---
        logger.info("\n--- 2. Transformer Modell tanítása ---")
        run_training(df_processed, logger)
    else:
        logger.error("A tanítási folyamat nem indul el, mert nem sikerült adatokat betölteni.")
    
    logger.info("=================================================")
    logger.info("            Folyamat sikeresen befejeződött.           ")
    logger.info("=================================================")


if __name__ == '__main__':
    main()
