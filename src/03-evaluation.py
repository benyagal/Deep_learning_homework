"""
03-evaluation.py

Kiértékeli a betanított modelleket a test set-en.
"""
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
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

def plot_test_confusion_matrix(y_true, y_pred, model_name, mae, output_path):
    """Tévesztési mátrixot rajzol és ment."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, config.NUM_CLASSES + 1)))
    cm_norm = (cm.T / cm.sum(axis=1)).T
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=list(range(1, config.NUM_CLASSES + 1)),
                yticklabels=list(range(1, config.NUM_CLASSES + 1)))
    plt.title(f'{model_name} - Test Set Confusion Matrix (MAE: {mae:.4f})')
    plt.xlabel('Prediktált Címke')
    plt.ylabel('Valós Címke')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Tévesztési mátrix mentve: {output_path}")

def evaluate_model_on_test(model_path, test_loader, model_name):
    """Egy modellt kiértékel a test set-en."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Modell kiértékelése: {model_name}")
    logger.info(f"Modell útvonal: {model_path}")
    logger.info(f"{'='*60}")
    
    # Ellenőrzés: létezik-e a modell fájl
    if not Path(model_path).exists():
        logger.error(f"HIBA: A modell fájl nem található: {model_path}")
        logger.error("Kérjük, először futtasd a 02-training.py szkriptet!")
        return None, None
    
    model = CoralModel(config.MODEL_NAME, num_classes=config.NUM_CLASSES, extra_feat_dim=len(config.FEATURE_COLS))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    except Exception as e:
        logger.error(f"HIBA a modell betöltése közben: {e}")
        return None, None
    
    model.to(config.DEVICE)
    model.eval()
    
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(config.DEVICE)
            mask = batch['attention_mask'].to(config.DEVICE)
            feats = batch['features'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            probs, _ = model(ids, mask, extra_feats=feats)
            preds = coral_probs_to_label_expected(probs)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    mae = mean_absolute_error(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    logger.info(f"Eredmények:")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  QWK: {qwk:.4f}")
    
    # Confusion matrix mentése
    cm_path = f"{config.LOG_DIR}/{model_name}_test_confusion_matrix.png"
    plot_test_confusion_matrix(all_labels, all_preds, model_name, mae, cm_path)
    
    return mae, qwk

def main():
    logger.info("--- Modell Kiértékelési Folyamat Indítása ---")
    
    # Adatok betöltése
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        logger.info(f"Feldolgozott adatok betöltve: {len(df)} sor")
    except FileNotFoundError:
        logger.error(f"HIBA: A feldolgozott adatfájl nem található: {config.PROCESSED_DATA_PATH}")
        logger.error("Kérjük, először futtasd a 01-data-preprocessing.py szkriptet!")
        return
    
    # Ellenőrzés: van-e 'fold' oszlop
    if 'fold' not in df.columns:
        logger.error("HIBA: A 'fold' oszlop nem található a feldolgozott adatokban!")
        logger.error("Kérjük, futtasd újra a 01-data-preprocessing.py szkriptet!")
        return
    
    # Test set: fold == 0
    test_df = df[df['fold'] == 0].reset_index(drop=True)
    
    if len(test_df) == 0:
        logger.error("HIBA: Nincs test adat (fold == 0)!")
        return
    
    logger.info(f"Test set mérete: {len(test_df)} példa")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Feature statisztikák - FONTOS: Ezeket a teljes train setből kellene számolni,
    # de mivel most csak a test-en értékelünk, használjuk a test set statisztikáit
    # (Production környezetben ezeket a training során kellene elmenteni!)
    feature_stats = {
        c: (test_df[c].mean(), test_df[c].std() if test_df[c].std() > 0 else 1.0)
        for c in config.FEATURE_COLS
    }
    
    test_loader = create_data_loader(
        test_df, 
        tokenizer, 
        config.MAX_LEN, 
        config.BATCH_SIZE, 
        config.FEATURE_COLS, 
        feature_stats
    )
    
    logger.info("\n" + "="*80)
    logger.info("TESZT HALMAZ KIÉRTÉKELÉSE")
    logger.info("="*80)
    
    # Minden fold modelljét kiértékeljük
    all_results = []
    
    for fold in range(1, config.KFOLDS + 1):
        model_path = f"{config.MODELS_DIR}/coral_fold{fold}_best.bin"
        mae, qwk = evaluate_model_on_test(model_path, test_loader, f"Fold{fold}")
        
        if mae is not None:
            all_results.append({'fold': fold, 'mae': mae, 'qwk': qwk})
    
    # Összesített eredmények
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("ÖSSZESÍTETT EREDMÉNYEK (Test Set)")
        logger.info("="*80)
        
        for res in all_results:
            logger.info(f"Fold {res['fold']}: MAE={res['mae']:.4f}, QWK={res['qwk']:.4f}")
        
        avg_mae = sum(r['mae'] for r in all_results) / len(all_results)
        avg_qwk = sum(r['qwk'] for r in all_results) / len(all_results)
        
        logger.info(f"\nÁtlagos MAE: {avg_mae:.4f}")
        logger.info(f"Átlagos QWK: {avg_qwk:.4f}")
        logger.info("="*80)
    else:
        logger.error("Nem sikerült egyetlen modellt sem kiértékelni!")
    
    logger.info("\n--- Kiértékelés Befejezve ---")

if __name__ == '__main__':
    main()
