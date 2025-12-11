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

from . import config
from .model import CoralModel, create_data_loader
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
    plt.title(f'Fold {fold} - Normalizált Tévesztési Mátrix (Best MAE: {best_mae:.4f})')
    plt.xlabel('Prediktált Címke')
    plt.ylabel('Valós Címke')
    
    # Mentés a log könyvtárba
    output_path = f"{config.LOG_DIR}/fold_{fold}_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Tévesztési mátrix mentve: {output_path}")

def run_training(df_processed, logger):
    """
    A teljes K-Fold keresztvalidációs tanítási folyamatot futtatja.
    """
    if df_processed.empty:
        logger.error("Nincs feldolgozott adat a tanításhoz.")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # --- Modell architektúra naplózása ---
    logger.info("--- Modell Architektúra ---")
    model_for_summary = CoralModel(config.MODEL_NAME, num_classes=config.NUM_CLASSES, extra_feat_dim=len(config.FEATURE_COLS))
    total_params = sum(p.numel() for p in model_for_summary.parameters())
    trainable_params = sum(p.numel() for p in model_for_summary.parameters() if p.requires_grad)
    logger.info(f"Modell: {config.MODEL_NAME} + CoralHead")
    logger.info(f"Összes paraméter: {total_params:,}")
    logger.info(f"Tanítható paraméter: {trainable_params:,}")
    logger.info("--------------------------")

    skf = StratifiedKFold(n_splits=config.KFOLDS, shuffle=True, random_state=config.SEED)
    labels_all = df_processed['label_int'].values
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_processed, labels_all), start=1):
        logger.info(f'\n[Fold {fold}/{config.KFOLDS}] Train size={len(train_idx)} Val size={len(val_idx)}')
        
        df_train_f = df_processed.iloc[train_idx].reset_index(drop=True)
        df_val_f = df_processed.iloc[val_idx].reset_index(drop=True)

        # Jellemzők normalizálásához szükséges statisztikák számítása (csak a train adatokon!)
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
            # A tqdm a loggerrel nem működik jól, ezért a standard print marad
            for batch in tqdm(train_loader_f, desc=f"Fold {fold} Epoch {epoch+1} Train"):
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
                torch.save(model_f.state_dict(), f'{config.MODEL_DIR}/coral_fold{fold}_best.bin')
                logger.info(f"Új legjobb modell mentve (MAE: {best_mae_f:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.PATIENCE:
                    logger.info(f'Early stopping a {epoch+1}. epoch-ban.')
                    break
        
        logger.info(f'Fold {fold} legjobb MAE: {best_mae_f:.4f}')
        fold_results.append(best_mae_f)

        # Legjobb modell betöltése és tévesztési mátrix készítése
        best_model_fold = CoralModel(config.MODEL_NAME, num_classes=config.NUM_CLASSES, extra_feat_dim=len(config.FEATURE_COLS))
        best_model_fold.load_state_dict(torch.load(f'{config.MODEL_DIR}/coral_fold{fold}_best.bin'))
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
    logger.info(f'\n--- Végső Kiértékelés ---')
    logger.info(f'Fold MAE-k: {fold_results}')
    logger.info(f'Átlagos MAE a {config.KFOLDS} foldon: {mean_mae:.4f}')
    logger.info("--------------------------")
