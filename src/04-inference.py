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
    
    # 2. Legjobb modell kiválasztása és betöltése
    import json
    best_model_info_path = f"{config.DATA_DIR}/best_model_info.json"
    
    # Próbáljuk betölteni a legjobb fold információt
    if Path(best_model_info_path).exists():
        with open(best_model_info_path, 'r') as f:
            best_info = json.load(f)
        best_fold = best_info['best_fold']
        logger.info(f"Legjobb modell: Fold {best_fold} (MAE={best_info['mae']:.4f}, QWK={best_info['qwk']:.4f})")
    else:
        logger.warning(f"Legjobb modell info nem talalhato: {best_model_info_path}")
        logger.warning("Fallback: Fold 5 modell hasznalata")
        best_fold = 5
    
    model_path = f"{config.MODELS_DIR}/coral_fold{best_fold}_best.bin"
    
    if not Path(model_path).exists():
        logger.error(f"HIBA: A modell fajl nem talalhato: {model_path}")
        logger.error("Kerlek, eloszor futtasd a 02-training.py es 03-evaluation.py szkripteket!")
        return
    
    logger.info(f"Modell betoltese: {model_path}")
    
    model = CoralModel(config.MODEL_NAME, num_classes=config.NUM_CLASSES, extra_feat_dim=len(config.FEATURE_COLS))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    except Exception as e:
        logger.error(f"HIBA a modell betoltese kozben: {e}")
        return
    
    model.to(config.DEVICE)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # 3. Feature normalizálás - betöltés a MENTETT statisztikákból (legjobb fold-hoz)
    stats_path = f"{config.MODELS_DIR}/coral_fold{best_fold}_feature_stats.json"
    
    try:
        with open(stats_path, 'r') as f:
            feature_stats = json.load(f)
        logger.info(f"Feature stats betoltve: {stats_path}")
        logger.info("(Ugyanazok a statisztikak, mint a Fold 5 training soran)")
    except FileNotFoundError:
        logger.warning(f"Feature stats nem talalhato: {stats_path}")
        logger.warning("Fallback: ujraszamolas a teljes processed_data.csv-bol")
        try:
            train_df = pd.read_csv(config.PROCESSED_DATA_PATH)
            feature_stats = {
                c: (train_df[c].mean(), train_df[c].std() if train_df[c].std() > 0 else 1.0)
                for c in config.FEATURE_COLS
            }
            logger.info("Feature stats ujraszamolva (lehet elteres!)")
        except Exception as e:
            logger.error(f"HIBA: {e}")
            logger.error("Feature normalizalas nelkul folytatom (NEM AJANLOTT!)")
            feature_stats = {c: (0.0, 1.0) for c in config.FEATURE_COLS}
    
    # 4. DataLoader létrehozása
    holdout_loader = create_data_loader(
        holdout_df,
        tokenizer,
        config.MAX_LEN,
        batch_size=1,  # Egyesével dolgozzuk fel
        feature_cols=config.FEATURE_COLS,
        feature_stats=feature_stats
    )
    
    # 5. Predikciók és eredmények
    logger.info("\n" + "-"*80)
    logger.info("PREDIKCIÓK:")
    logger.info("-"*80)
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for idx, batch in enumerate(holdout_loader):
            ids = batch['input_ids'].to(config.DEVICE)
            mask = batch['attention_mask'].to(config.DEVICE)
            feats = batch['features'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            probs, _ = model(ids, mask, extra_feats=feats)
            preds = coral_probs_to_label_expected(probs)
            
            true_label = int(labels.item())
            pred_label = int(preds.item())
            
            all_labels.append(true_label)
            all_preds.append(pred_label)
            
            text_snippet = holdout_df.iloc[idx]['paragraph_text'][:100]
            logger.info(f"\nPelda #{idx+1}:")
            logger.info(f"  Szoveg: {text_snippet}...")
            logger.info(f"  Valos cimke: {true_label}")
            logger.info(f"  Prediktalt cimke: {pred_label}")
            logger.info(f"  Egyezes: {'IGEN' if true_label == pred_label else 'NEM'}")
    
    # 6. Osszesitett metrikak
    logger.info("\n" + "="*80)
    logger.info("OSSZESITETT EREDMENYEK (Holdout Set)")
    logger.info("="*80)
    
    mae = mean_absolute_error(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    accuracy = sum(1 for t, p in zip(all_labels, all_preds) if t == p) / len(all_labels)
    
    logger.info(f"Holdout set merete: {len(holdout_df)} pelda")
    logger.info(f"MAE (Mean Absolute Error): {mae:.4f}")
    logger.info(f"QWK (Quadratic Weighted Kappa): {qwk:.4f}")
    logger.info(f"Pontossag: {accuracy:.2%}")
    logger.info("="*80)
    
    logger.info("\n--- Inference Befejezve ---")

if __name__ == '__main__':
    run_inference_on_holdout()
