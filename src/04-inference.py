"""
04-inference.py

Inference a kiválasztott holdout adatokon (új, nem látott példák).
"""
import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from pathlib import Path

import config
from model import CoralModel, create_data_loader
from utils import get_logger

logger = get_logger(__name__)

def coral_probs_to_label_expected(probs):
    """Várható érték számítása a CORAL valószínűségekből."""
    B = probs.size(0)
    ones = torch.ones(B, 1, device=probs.device)
    zeros = torch.zeros(B, 1, device=probs.device)
    p_gt = torch.cat([ones, probs, zeros], dim=1)
    p_exact = p_gt[:, :-1] - p_gt[:, 1:]
    labels = torch.arange(1, probs.size(1) + 2, device=probs.device).float()
    exp_val = torch.sum(p_exact * labels, dim=1)
    return torch.clamp(torch.round(exp_val), 1, probs.size(1) + 1).long()

def run_inference_on_holdout():
    """
    Inference futtatása a holdout adatokon.
    Ez a holdout set kizárólag olyan példákat tartalmaz, amelyeket a modell
    tanítás során SOHA nem látott.
    """
    logger.info("\n" + "="*80)
    logger.info("INFERENCE HOLDOUT ADATOKON (Uj, Nem Latott Peldak)")
    logger.info("="*80)
    
    # 1. Holdout adatok betöltése
    holdout_path = config.INFERENCE_HOLDOUT_PATH
    
    if not Path(holdout_path).exists():
        logger.error(f"HIBA: A holdout fajl nem talalhato: {holdout_path}")
        logger.error("Kerlek, eloszor futtasd a 01-data-preprocessing.py szkriptet!")
        return
    
    try:
        holdout_df = pd.read_csv(holdout_path)
        logger.info(f"Holdout adatok betoltve: {len(holdout_df)} pelda")
    except Exception as e:
        logger.error(f"HIBA a holdout adatok betoltese kozben: {e}")
        return
    
    # Ellenorzes: szukseges oszlopok meglete
    required_cols = ['paragraph_text', 'label_int'] + config.FEATURE_COLS
    missing_cols = [col for col in required_cols if col not in holdout_df.columns]
    if missing_cols:
        logger.error(f"HIBA: Hianyzoo oszlopok a holdout adatokban: {missing_cols}")
        return
    
    # 2. ENSEMBLE PREDICTION - Mind az 5 fold modellje
    import json
    
    logger.info("\n" + "="*80)
    logger.info("ENSEMBLE PREDICTION (5 fold modellek atlaga)")
    logger.info("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Minden fold predikcioit taroljuk
    all_fold_predictions = []
    
    for fold in range(1, config.KFOLDS + 1):
        model_path = f"{config.MODELS_DIR}/coral_fold{fold}_best.bin"
        stats_path = f"{config.MODELS_DIR}/coral_fold{fold}_feature_stats.json"
        
        # Modell ellenorzes
        if not Path(model_path).exists():
            logger.warning(f"Fold {fold} modell nem talalhato: {model_path}")
            logger.warning(f"Kihagyva a fold {fold} az ensemble-bol")
            continue
        
        # Feature stats betoltese
        if Path(stats_path).exists():
            with open(stats_path, 'r') as f:
                feature_stats = json.load(f)
        else:
            logger.warning(f"Fold {fold} feature stats nem talalhato, ujraszamolas")
            train_df = pd.read_csv(config.PROCESSED_DATA_PATH)
            feature_stats = {
                c: (train_df[c].mean(), train_df[c].std() if train_df[c].std() > 0 else 1.0)
                for c in config.FEATURE_COLS
            }
        
        # Modell betoltese
        model = CoralModel(config.MODEL_NAME, num_classes=config.NUM_CLASSES, extra_feat_dim=len(config.FEATURE_COLS))
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()
        
        # DataLoader letrehozasa fold-specifikus feature stats-szal
        holdout_loader = create_data_loader(
            holdout_df,
            tokenizer,
            config.MAX_LEN,
            batch_size=1,
            feature_cols=config.FEATURE_COLS,
            feature_stats=feature_stats
        )
        
        # Fold predikciok
        fold_preds = []
        with torch.no_grad():
            for batch in holdout_loader:
                ids = batch['input_ids'].to(config.DEVICE)
                mask = batch['attention_mask'].to(config.DEVICE)
                feats = batch['features'].to(config.DEVICE)
                
                probs, _ = model(ids, mask, extra_feats=feats)
                preds = coral_probs_to_label_expected(probs)
                fold_preds.append(int(preds.item()))
        
        all_fold_predictions.append(fold_preds)
        logger.info(f"Fold {fold} predikciok: {fold_preds}")
    
    if len(all_fold_predictions) == 0:
        logger.error("HIBA: Egyetlen modell sem toltheto be!")
        return
    
    # Ensemble: atlagolas es kerekites
    all_fold_predictions = np.array(all_fold_predictions)  # Shape: (n_folds, n_samples)
    ensemble_preds_raw = np.mean(all_fold_predictions, axis=0)  # Atlag
    ensemble_preds = np.round(ensemble_preds_raw).astype(int)  # Kerekites
    ensemble_preds = np.clip(ensemble_preds, 1, config.NUM_CLASSES)  # Biztonsag
    
    logger.info(f"\nEnsemble atlag (kerekites elott): {ensemble_preds_raw}")
    logger.info(f"Vegleges ensemble predikciok: {list(ensemble_preds)}")
    logger.info("="*80)
    
    # 3. Eredmenyek megjelenites
    logger.info("\n" + "-"*80)
    logger.info("PREDIKCIÓK (Ensemble):")
    logger.info("-"*80)
    
    all_labels = []
    all_preds = []
    
    for idx in range(len(holdout_df)):
        true_label = int(holdout_df.iloc[idx]['label_int'])
        pred_label = int(ensemble_preds[idx])
        
        all_labels.append(true_label)
        all_preds.append(pred_label)
        
        text_snippet = holdout_df.iloc[idx]['paragraph_text'][:100]
        logger.info(f"\nPelda #{idx+1}:")
        logger.info(f"  Szoveg: {text_snippet}...")
        logger.info(f"  Valos cimke: {true_label}")
        logger.info(f"  Ensemble predikciok: {all_fold_predictions[:, idx]}")
        logger.info(f"  Vegleges predikalt cimke: {pred_label}")
        logger.info(f"  Egyezes: {'IGEN' if true_label == pred_label else 'NEM'}")
    
    # 4. Osszesitett metrikak
    logger.info("\n" + "="*80)
    logger.info("OSSZESITETT EREDMENYEK (Holdout Set - Ensemble)")
    logger.info("="*80)
    
    mae = mean_absolute_error(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    accuracy = sum(1 for t, p in zip(all_labels, all_preds) if t == p) / len(all_labels)
    
    logger.info(f"Holdout set merete: {len(holdout_df)} pelda")
    logger.info(f"Hasznalt modellek szama: {len(all_fold_predictions)}")
    logger.info(f"MAE (Mean Absolute Error): {mae:.4f}")
    logger.info(f"QWK (Quadratic Weighted Kappa): {qwk:.4f}")
    logger.info(f"Pontossag: {accuracy:.2%}")
    logger.info("="*80)
    
    logger.info("\n--- Inference Befejezve ---")

if __name__ == '__main__':
    run_inference_on_holdout()
