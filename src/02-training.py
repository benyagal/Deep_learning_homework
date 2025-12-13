"""
Tanítási és kiértékelési funkciók.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, cohen_kappa_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mord import LogisticAT  # Ordinal regression baseline

import config
from model import CoralModel, create_data_loader
from transformers import AutoTokenizer

def coral_loss_simple(probs, labels):
    """Egyszerűsített CORAL veszteségfüggvény."""
    K = probs.size(1) + 1
    targets = []
    for k in range(1, K):
        target_k = (labels > k).float().unsqueeze(1)
        targets.append(target_k)
    target_tensor = torch.cat(targets, dim=1)
    return nn.BCELoss()(probs, target_tensor)

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

def plot_confusion_matrix(y_true, y_pred, fold, best_mae, logger):
    """Tévesztési mátrixot rajzol és ment."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, config.NUM_CLASSES + 1)))
    cm_norm = (cm.T / cm.sum(axis=1)).T  # Normalizálás a valós címkék szerint
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=list(range(1, config.NUM_CLASSES + 1)), 
                yticklabels=list(range(1, config.NUM_CLASSES + 1)))
    plt.title(f'Fold {fold} - Normalizalt Tevesztesi Matrix (Best MAE: {best_mae:.4f})')
    plt.xlabel('Prediktalt Cimke')
    plt.ylabel('Valos Cimke')
    
    # Mentes a log konyvtarba
    output_path = f"{config.LOG_DIR}/fold_{fold}_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Tevesztesi matrix mentve: {output_path}")

def train_baseline_model(df_processed, logger):
    """
    Baseline modell: LogisticAT ordinal regression csak a 23 handcrafted feature-rel.
    Egyszeri tanitas 80/20 train/test split-tel.
    """
    from sklearn.model_selection import train_test_split
    
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE MODEL - Ordinal Regression (LogisticAT)")
    logger.info("=" * 80)
    logger.info("Csak a 23 handcrafted feature-t hasznalja (nincs transformer)")
    logger.info(f"Modell: mord.LogisticAT (Ordinal Logistic Regression)")
    logger.info(f"Features: {len(config.FEATURE_COLS)} db")
    logger.info(f"Train/Test split: 80/20")
    logger.info("=" * 80 + "\n")
    
    # Train/test split (80/20)
    X = df_processed[config.FEATURE_COLS].values
    y = df_processed['label_int'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} pelda")
    logger.info(f"Test set:  {len(X_test)} pelda")
    
    # LogisticAT tanítása
    logger.info("\nTanitas...")
    baseline_model = LogisticAT(alpha=1.0)  # Ridge regularization
    baseline_model.fit(X_train, y_train)
    
    # Predikciók és metrikák
    y_pred = baseline_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    
    logger.info("\n" + "-" * 80)
    logger.info("BASELINE EREDMENYEK:")
    logger.info(f"  Test MAE: {mae:.4f}")
    logger.info(f"  Test QWK: {qwk:.4f}")
    logger.info("  (Ez a referenciaeredmeny a deep learning modellhez kepest)")
    logger.info("-" * 80 + "\n")
    
    return mae, qwk

def run_training(df_processed, logger):
    """
    A teljes K-Fold keresztvalidációs tanítási folyamatot futtatja.
    Először a baseline modell, utána a deep learning modell.
    """
    if df_processed.empty:
        logger.error("Nincs feldolgozott adat a tanitashoz.")
        return

    # === 1. BASELINE MODELL ===
    baseline_mae, baseline_qwk = train_baseline_model(df_processed, logger)
    
    # === 2. DEEP LEARNING MODELL ===
    logger.info("\n" + "=" * 80)
    logger.info("DEEP LEARNING MODEL - CORAL + Transformer")
    logger.info("=" * 80)
    logger.info("Hibrid modell: Transformer embeddings + 23 handcrafted features")
    logger.info("=" * 80 + "\n")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # --- Modell architektura naplozasa ---
    logger.info("\n" + "=" * 80)
    logger.info("MODELL ARCHITEKTURA")
    logger.info("=" * 80)
    model_for_summary = CoralModel(config.MODEL_NAME, num_classes=config.NUM_CLASSES, extra_feat_dim=len(config.FEATURE_COLS))
    
    # Backbone paraméterek
    backbone_params = sum(p.numel() for p in model_for_summary.base.parameters())
    backbone_trainable = sum(p.numel() for p in model_for_summary.base.parameters() if p.requires_grad)
    
    # Head paraméterek
    head_params = sum(p.numel() for p in model_for_summary.head.parameters())
    head_trainable = sum(p.numel() for p in model_for_summary.head.parameters() if p.requires_grad)
    
    # Teljes paraméterek
    total_params = backbone_params + head_params
    trainable_params = backbone_trainable + head_trainable
    
    logger.info(f"\nTransformer Backbone: {config.MODEL_NAME}")
    logger.info(f"  - Backbone parameterek: {backbone_params:,} (tanitható: {backbone_trainable:,})")
    logger.info(f"  - Hidden size: 768")
    logger.info(f"  - Layers: 12 (transformer encoder)")
    
    logger.info(f"\nCORAL Head:")
    logger.info(f"  - Input: [CLS] (768) + Extra Features ({len(config.FEATURE_COLS)})")
    logger.info(f"  - MLP hidden layer: 256 (LayerNorm + GELU + Dropout)")
    logger.info(f"  - Output: {config.NUM_CLASSES - 1} kuszob (sigmoid)")
    logger.info(f"  - Head parameterek: {head_params:,} (mind tanitható)")
    
    logger.info(f"\nOsszesen:")
    logger.info(f"  - Osszes parameter: {total_params:,}")
    logger.info(f"  - Tanitható parameter: {trainable_params:,}")
    logger.info(f"  - Nem-tanitható: {total_params - trainable_params:,}")
    logger.info("=" * 80 + "\n")

    skf = StratifiedKFold(n_splits=config.KFOLDS, shuffle=True, random_state=config.SEED)
    labels_all = df_processed['label_int'].values
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_processed, labels_all), start=1):
        logger.info(f'\n[Fold {fold}/{config.KFOLDS}] Train size={len(train_idx)} Val size={len(val_idx)}')
        
        df_train_f = df_processed.iloc[train_idx].reset_index(drop=True)
        df_val_f = df_processed.iloc[val_idx].reset_index(drop=True)

        # Jellemzok normalizalasahoz szukseges statisztikak szamitasa (csak a train adatokon!)
        feature_stats_f = {
            c: (df_train_f[c].mean(), df_train_f[c].std() if df_train_f[c].std() > 0 else 1.0) 
            for c in config.FEATURE_COLS
        }

        train_loader_f = create_data_loader(df_train_f, tokenizer, config.MAX_LEN, config.BATCH_SIZE, config.FEATURE_COLS, feature_stats_f)
        val_loader_f = create_data_loader(df_val_f, tokenizer, config.MAX_LEN, config.BATCH_SIZE, config.FEATURE_COLS, feature_stats_f)

        model_f = CoralModel(config.MODEL_NAME, num_classes=config.NUM_CLASSES, extra_feat_dim=len(config.FEATURE_COLS)).to(config.DEVICE)
        optimizer_f = torch.optim.AdamW(model_f.parameters(), lr=config.LEARNING_RATE)
        scheduler_f = CosineAnnealingWarmRestarts(optimizer_f, T_0=config.EPOCHS, T_mult=1, eta_min=1e-7)

        best_mae_f = float('inf')
        epochs_no_improve = 0

        for epoch in range(config.EPOCHS):
            model_f.train()
            train_losses = []
            # Nincs tqdm - tiszta, tömör log
            for batch in train_loader_f:
                ids = batch['input_ids'].to(config.DEVICE)
                mask = batch['attention_mask'].to(config.DEVICE)
                feats = batch['features'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)

                probs, _ = model_f(ids, mask, extra_feats=feats)
                loss = coral_loss_simple(probs, labels)
                train_losses.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(model_f.parameters(), 1.0)
                optimizer_f.step()
                optimizer_f.zero_grad()

            scheduler_f.step()

            # Validáció
            model_f.eval()
            all_lab, all_pred, val_losses = [], [], []
            with torch.no_grad():
                for batch in val_loader_f: # Nincs tqdm a validációnál a tiszta log érdekében
                    ids = batch['input_ids'].to(config.DEVICE)
                    mask = batch['attention_mask'].to(config.DEVICE)
                    feats = batch['features'].to(config.DEVICE)
                    labels = batch['labels'].to(config.DEVICE)

                    probs, _ = model_f(ids, mask, extra_feats=feats)
                    val_loss = coral_loss_simple(probs, labels)
                    val_losses.append(val_loss.item())

                    preds = coral_probs_to_label_expected(probs)
                    all_lab.extend(labels.cpu().numpy())
                    all_pred.extend(preds.cpu().numpy())

            mae_f = mean_absolute_error(all_lab, all_pred)
            qwk_f = cohen_kappa_score(all_lab, all_pred, weights='quadratic')
            logger.info(f'Epoch {epoch+1}: TrainLoss={np.mean(train_losses):.4f} ValLoss={np.mean(val_losses):.4f} ValMAE={mae_f:.4f} QWK={qwk_f:.4f}')

            if mae_f < best_mae_f:
                best_mae_f = mae_f
                epochs_no_improve = 0
                torch.save(model_f.state_dict(), f'{config.MODELS_DIR}/coral_fold{fold}_best.bin')
                logger.info(f"Uj legjobb modell mentve (MAE: {best_mae_f:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.PATIENCE:
                    logger.info(f'Early stopping a {epoch+1}. epoch-ban.')
                    break
        
        logger.info(f'Fold {fold} legjobb MAE: {best_mae_f:.4f}')
        fold_results.append(best_mae_f)

        # Legjobb modell betöltése és tévesztési mátrix készítése
        best_model_fold = CoralModel(config.MODEL_NAME, num_classes=config.NUM_CLASSES, extra_feat_dim=len(config.FEATURE_COLS))
        best_model_fold.load_state_dict(torch.load(f'{config.MODELS_DIR}/coral_fold{fold}_best.bin'))
        best_model_fold.to(config.DEVICE)
        best_model_fold.eval()
        
        all_lab_best, all_pred_best = [], []
        with torch.no_grad():
            for batch in val_loader_f:
                ids = batch['input_ids'].to(config.DEVICE)
                mask = batch['attention_mask'].to(config.DEVICE)
                feats = batch['features'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)
                probs, _ = best_model_fold(ids, mask, extra_feats=feats)
                preds = coral_probs_to_label_expected(probs)
                all_lab_best.extend(labels.cpu().numpy())
                all_pred_best.extend(preds.cpu().numpy())
        
        plot_confusion_matrix(all_lab_best, all_pred_best, fold, best_mae_f, logger)

    mean_mae = np.mean(fold_results)
    logger.info(f'\n--- Vegso Kiertekeles ---')
    logger.info(f'Fold MAE-k: {fold_results}')
    logger.info(f'Atlagos MAE a {config.KFOLDS} foldon: {mean_mae:.4f}')
    logger.info("--------------------------")
    
    # === OSSZEHASONLITAS ===
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE vs DEEP LEARNING OSSZEHASONLITAS")
    logger.info("=" * 80)
    logger.info(f"Baseline (LogisticAT):        MAE = {baseline_mae:.4f}")
    logger.info(f"Deep Learning (CORAL+BERT):   MAE = {mean_mae:.4f}")
    improvement = ((baseline_mae - mean_mae) / baseline_mae) * 100
    logger.info(f"Javulas: {improvement:.1f}%")
    logger.info("=" * 80 + "\n")

if __name__ == '__main__':
    """Standalone futtatás: CSV betöltése és tanítás indítása."""
    import pandas as pd
    from utils import get_logger
    
    logger = get_logger(__name__)
    logger.info("02-training.py: Standalone mod inditasa")
    
    # Adatok betoltese
    logger.info(f"Feldolgozott adatok betoltese: {config.PROCESSED_DATA_PATH}")
    try:
        df_processed = pd.read_csv(config.PROCESSED_DATA_PATH)
        logger.info(f"Betoltve {len(df_processed)} sor, {len(df_processed.columns)} oszlop")
        logger.info(f"Oszlopok: {list(df_processed.columns)}")
    except FileNotFoundError:
        logger.error(f"HIBA: A feldolgozott adatfajl nem talalhato: {config.PROCESSED_DATA_PATH}")
        logger.error("Kerlek, eloszor futtasd a 01-data-preprocessing.py szkriptet!")
        exit(1)
    
    # Tanítás futtatása
    run_training(df_processed, logger)
