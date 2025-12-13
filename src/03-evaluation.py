"""
03-evaluation.py

A betanított modellek kiértékelését végző szkript.
Betölti a tesztadatokat és a mentett modelleket, majd kiszámolja
és naplózza a teljesítménymutatókat (MAE, QWK).
"""
import logging
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from mord import LogisticAT
from transformers import AutoModel, AutoTokenizer

from config import (
    PROCESSED_DATA_PATH, MODELS_DIR, MODEL_NAME, BASELINE_MODEL_NAME,
    TARGET_COL, TEXT_COL, FEATURE_COLS, NUM_CLASSES, MAX_LEN
)
from model import CoralModel, create_data_loader
from utils import get_logger

logger = get_logger(__name__)

def evaluate_baseline(test_df: pd.DataFrame, model_path: Path, logger: logging.Logger):
    """A baseline modell kiértékelése."""
    logger.info(f"Baseline modell kiértékelése: {model_path}")
    try:
        baseline_model = joblib.load(model_path)
    except FileNotFoundError:
        logger.error(f"Baseline modell nem található: {model_path}")
        return

    feature_cols = [col for col in test_df.columns if col not in [TARGET_COL, TEXT_COL, 'task_id', 'paragraph_text', 'label_text']]
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values

    y_pred = baseline_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')

    logger.info("--- Baseline Modell Eredmények ---")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"QWK: {qwk:.4f}")
    logger.info("------------------------------------")

def evaluate_transformer_model(test_df: pd.DataFrame, model_path: Path, logger: logging.Logger):
    """A transzformer alapú modell kiértékelése."""
    logger.info(f"Transzformer modell kiértékelése: {model_path}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA nem elérhető, a kiértékelés CPU-n fut. Ez lassú lehet.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    try:
        # Modell architektúra újbóli létrehozása - same as training
        model = CoralModel(MODEL_NAME, num_classes=NUM_CLASSES, extra_feat_dim=len(FEATURE_COLS))
        
        # Mentett súlyok betöltése
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        logger.error(f"Transzformer modell nem található: {model_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare feature statistics from train data (simplified: use test stats for now)
    feature_stats = {
        c: (test_df[c].mean(), test_df[c].std() if test_df[c].std() > 0 else 1.0)
        for c in FEATURE_COLS
    }
    
    # Create dataloader for batch processing
    test_loader = create_data_loader(test_df, tokenizer, MAX_LEN, batch_size=8, 
                                      feature_cols=FEATURE_COLS, feature_stats=feature_stats)
    
    y_test = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            feats = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            probs, _ = model(ids, mask, extra_feats=feats)
            
            # CORAL prediction: expected value from probabilities
            B = probs.size(0)
            ones = torch.ones(B, 1, device=device)
            zeros = torch.zeros(B, 1, device=device)
            p_gt = torch.cat([ones, probs, zeros], dim=1)
            p_exact = p_gt[:, :-1] - p_gt[:, 1:]
            label_values = torch.arange(1, NUM_CLASSES + 1, device=device).float()
            exp_val = torch.sum(p_exact * label_values, dim=1)
            preds = torch.clamp(torch.round(exp_val), 1, NUM_CLASSES).long()
            
            y_test.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    mae = mean_absolute_error(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')

    logger.info("--- Transzformer Modell Eredmények ---")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"QWK: {qwk:.4f}")
    logger.info("------------------------------------")


def main():
    """Fő kiértékelési funkció."""
    logger.info("--- Modell Kiértékelési Folyamat Indítása ---")

    # Adatok betöltése
    processed_data_file = Path(PROCESSED_DATA_PATH)
    if not processed_data_file.exists():
        logger.error(f"Előfeldolgozott adatfájl nem található: {PROCESSED_DATA_PATH}. Kérlek, futtasd először a 01-es és 02-es szkripteket.")
        return
    
    df = pd.read_csv(processed_data_file)
    
    # A K-Fold validáció miatt a "teszt" adat most az utolsó fold lesz.
    # Egy valós rendszerben dedikált teszt adathalmazt használnánk.
    # Itt most a 0-s fold-on tanított modelleket értékeljük ki a 0-s fold teszt adatain.
    test_df = df[df['fold'] == 0]
    
    # A fold 1-gyel kezdődik a training során (fold 1, 2, 3, 4, 5), de a fold oszlop 0-tól indul
    # Ezért a 0-s fold-on tanított modell a coral_fold1_best.bin
    test_fold_id = 0
    model_fold_id = test_fold_id + 1  # fold=0 teszt -> fold1 modell
    
    # Baseline modell kiértékelése (ha létezik)
    baseline_model_path = Path(MODELS_DIR) / f"{BASELINE_MODEL_NAME}_fold_{model_fold_id}.joblib"
    if baseline_model_path.exists():
        evaluate_baseline(test_df, baseline_model_path, logger)
    else:
        logger.warning(f"Baseline modell nem található: {baseline_model_path} (opcionális)")

    # Transzformer modell kiértékelése
    transformer_model_path = Path(MODELS_DIR) / f"coral_fold{model_fold_id}_best.bin"
    if transformer_model_path.exists():
        evaluate_transformer_model(test_df, transformer_model_path, logger)
    else:
        logger.error(f"Transzformer modell nem található: {transformer_model_path}")
        logger.error("Kérlek, futtasd először a 02-training.py szkriptet!")

    logger.info("--- Modell Kiértékelési Folyamat Befejeződött ---")

if __name__ == '__main__':
    main()
