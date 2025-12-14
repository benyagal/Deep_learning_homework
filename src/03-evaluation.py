"""
03-evaluation.py

Kiértékeli a baseline és deep learning modelleket a validation set-en.
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

def plot_validation_confusion_matrix(y_true, y_pred, model_name, mae, output_path):
    """Tévesztési mátrixot rajzol és ment."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, config.NUM_CLASSES + 1)))
    cm_norm = (cm.T / cm.sum(axis=1)).T
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=list(range(1, config.NUM_CLASSES + 1)),
                yticklabels=list(range(1, config.NUM_CLASSES + 1)))
    plt.title(f'{model_name} - Validation Set Confusion Matrix (MAE: {mae:.4f})')
    plt.xlabel('Prediktalt Cimke')
    plt.ylabel('Valos Cimke')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Tevesztesi matrix mentve: {output_path}")

def load_baseline_results():
    """Baseline eredmények betöltése (02-training.py futtatása után)."""
    import json
    
    logger.info("\n" + "="*80)
    logger.info("BASELINE MODEL RESULTS (betoltve)")
    logger.info("="*80)
    
    results_path = f"{config.DATA_DIR}/baseline_results.json"
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        mae = results['mae']
        qwk = results['qwk']
        
        logger.info(f"Baseline Test MAE: {mae:.4f}")
        logger.info(f"Baseline Test QWK: {qwk:.4f}")
        logger.info("(Eredmenyek a 02-training.py altal mentett fajlbol)")
        logger.info("="*80 + "\n")
        
        return mae, qwk
    
    except FileNotFoundError:
        logger.error(f"HIBA: Baseline eredmeny fajl nem talalhato: {results_path}")
        logger.error("Kerlek, eloszor futtasd a 02-training.py szkriptet!")
        return None, None

def evaluate_model_on_validation(model_path, val_loader, model_name):
    """Egy modellt kiértékel a validation set-en."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Modell kiertekeles: {model_name}")
    logger.info(f"Modell utvonal: {model_path}")
    logger.info(f"{'='*60}")
    
    # Ellenorzes: letezik-e a modell fajl
    if not Path(model_path).exists():
        logger.error(f"HIBA: A modell fajl nem talalhato: {model_path}")
        logger.error("Kerlek, eloszor futtasd a 02-training.py szkriptet!")
        return None, None
    
    model = CoralModel(config.MODEL_NAME, num_classes=config.NUM_CLASSES, extra_feat_dim=len(config.FEATURE_COLS))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    except Exception as e:
        logger.error(f"HIBA a modell betoltese kozben: {e}")
        return None, None
    
    model.to(config.DEVICE)
    model.eval()
    
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for batch in val_loader:
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
    
    logger.info(f"Eredmenyek:")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  QWK: {qwk:.4f}")
    
    # Confusion matrix mentese a data konyvtarba (mountolt terulet)
    cm_path = f"{config.DATA_DIR}/{model_name}_validation_confusion_matrix.png"
    plot_validation_confusion_matrix(all_labels, all_preds, model_name, mae, cm_path)
    
    return mae, qwk

def main():
    logger.info("--- Modell Kiertekeles Folyamat Inditasa ---")
    
    # Adatok betoltese
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        logger.info(f"Feldolgozott adatok betoltve: {len(df)} sor")
    except FileNotFoundError:
        logger.error(f"HIBA: A feldolgozott adatfajl nem talalhato: {config.PROCESSED_DATA_PATH}")
        logger.error("Kerlek, eloszor futtasd a 01-data-preprocessing.py szkriptet!")
        return
    
    # Ellenorzes: van-e 'fold' oszlop
    if 'fold' not in df.columns:
        logger.error("HIBA: A 'fold' oszlop nem talalhato a feldolgozott adatokban!")
        logger.error("Kerlek, futtasd ujra a 01-data-preprocessing.py szkriptet!")
        return
    
    # === 1. BASELINE MODEL RESULTS (betoltes) ===
    baseline_mae, baseline_qwk = load_baseline_results()
    
    if baseline_mae is None:
        logger.error("Baseline eredmenyek nem toltheto be. Kilepes.")
        return
    
    # === 2. DEEP LEARNING MODEL EVALUATION ===
    logger.info("\n" + "="*80)
    logger.info("DEEP LEARNING MODEL VALIDATION (5-Fold)")
    logger.info("="*80)
    logger.info("NOTE: Minden modell a sajat validation fold-jan van ertkelve.")
    logger.info("      (pl. Fold1 model -> validacio fold=0, Fold2 model -> validacio fold=1)\n")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    all_results = []
    
    # Minden fold iteráció: saját validation fold-on tesztelés
    for fold in range(config.KFOLDS):  # 0, 1, 2, 3, 4
        # Validation fold: ahol NEM tanítottunk
        val_df = df[df['fold'] == fold].reset_index(drop=True)
        
        if len(val_df) == 0:
            logger.warning(f"FIGYELEM: Nincs validation adat fold={fold}-ra!")
            continue
        
        # Modell fájl neve: fold+1 (mert coral_fold1_best.bin, coral_fold2_best.bin, stb.)
        model_path = f"{config.MODELS_DIR}/coral_fold{fold+1}_best.bin"
        model_name = f"Fold{fold+1}"
        
        logger.info(f"[{model_name}] Validation fold: {fold} ({len(val_df)} pelda)")
        
        # Feature stats betoltese (NEM ujraszamolas!)
        import json
        stats_path = f"{config.MODELS_DIR}/coral_fold{fold+1}_feature_stats.json"
        
        if Path(stats_path).exists():
            with open(stats_path, 'r') as f:
                feature_stats = json.load(f)
            logger.info(f"Feature stats betoltve: {stats_path}")
        else:
            logger.warning(f"Feature stats nem talalhato: {stats_path}")
            logger.warning("Ujraszamolas a train foldbol (lehetelteres a trainingtol!)")
            train_df = df[df['fold'] != fold]
            feature_stats = {
                c: (train_df[c].mean(), train_df[c].std() if train_df[c].std() > 0 else 1.0)
                for c in config.FEATURE_COLS
            }
        
        val_loader = create_data_loader(
            val_df, 
            tokenizer, 
            config.MAX_LEN, 
            config.BATCH_SIZE, 
            config.FEATURE_COLS, 
            feature_stats
        )
        
        mae, qwk = evaluate_model_on_validation(model_path, val_loader, model_name)
        
        if mae is not None:
            all_results.append({'fold': fold+1, 'mae': mae, 'qwk': qwk})
    
    # Osszesitett eredmenyek
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("OSSZESITETT EREDMENYEK (Validation Set)")
        logger.info("="*80)
        
        for res in all_results:
            logger.info(f"Fold {res['fold']}: MAE={res['mae']:.4f}, QWK={res['qwk']:.4f}")
        
        avg_mae = sum(r['mae'] for r in all_results) / len(all_results)
        avg_qwk = sum(r['qwk'] for r in all_results) / len(all_results)
        
        logger.info(f"\nAtlagos MAE: {avg_mae:.4f}")
        logger.info(f"Atlagos QWK: {avg_qwk:.4f}")
        
        # Legjobb fold kivalasztasa (legkisebb MAE)
        best_result = min(all_results, key=lambda x: x['mae'])
        best_fold = best_result['fold']
        logger.info(f"\nLegjobb fold: Fold {best_fold} (MAE={best_result['mae']:.4f}, QWK={best_result['qwk']:.4f})")
        
        # Legjobb fold mentese JSON-ba (04-inference.py-hoz)
        import json
        best_model_info = {
            'best_fold': best_fold,
            'mae': best_result['mae'],
            'qwk': best_result['qwk']
        }
        best_model_path = f"{config.DATA_DIR}/best_model_info.json"
        with open(best_model_path, 'w') as f:
            json.dump(best_model_info, f, indent=2)
        logger.info(f"Legjobb modell info mentve: {best_model_path}")
        logger.info("="*80)
        
        # === 3. BASELINE vs DEEP LEARNING COMPARISON ===
        logger.info("\n" + "="*80)
        logger.info("FINAL COMPARISON: BASELINE vs DEEP LEARNING")
        logger.info("="*80)
        logger.info(f"Baseline (LogisticAT):        MAE = {baseline_mae:.4f}, QWK = {baseline_qwk:.4f}")
        logger.info(f"Deep Learning (CORAL+BERT):   MAE = {avg_mae:.4f}, QWK = {avg_qwk:.4f}")
        improvement_mae = ((baseline_mae - avg_mae) / baseline_mae) * 100
        improvement_qwk = ((avg_qwk - baseline_qwk) / baseline_qwk) * 100
        logger.info(f"Javulas MAE: {improvement_mae:.1f}%")
        logger.info(f"Javulas QWK: {improvement_qwk:.1f}%")
        logger.info("="*80)
    else:
        logger.error("Nem sikerult egyetlen modellt sem kiertkelni!")
    
    logger.info("\n--- Kiertekeles Befejezve ---")

if __name__ == '__main__':
    main()
