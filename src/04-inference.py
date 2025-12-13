"""
04-inference.py

Szkript, amely egy betanított modellel végez predikciót a holdout (unseen) 
adathalmazon. Ezek a példák NEM szerepeltek a training/validation során.
"""
import torch
import argparse
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

from config import MODELS_DIR, MODEL_NAME, NUM_CLASSES, MAX_LEN, FEATURE_COLS, INFERENCE_HOLDOUT_PATH
from model import CoralModel
from utils import get_logger

logger = get_logger(__name__)

def run_inference_on_holdout(holdout_path: str, model_path: str):
    """
    Predikciót futtat a holdout (unseen) adathalmazon.
    """
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE HOLDOUT ADATOKON (UNSEEN DATA)")
    logger.info("=" * 80)
    logger.info(f"Holdout fájl: {holdout_path}")
    logger.info(f"Modell fájl: {model_path}")
    
    # Holdout adat betöltése
    holdout_file = Path(holdout_path)
    if not holdout_file.exists():
        logger.error(f"Holdout fájl nem található: {holdout_path}")
        logger.error("Kérlek, futtasd először a 01-data-preprocessing.py szkriptet!")
        return
    
    df_holdout = pd.read_csv(holdout_file)
    logger.info(f"Holdout minták száma: {len(df_holdout)}")
    logger.info(f"Oszlopok: {list(df_holdout.columns)[:5]}...")  # Első 5 oszlop
    
    # Device beállítás
    if not torch.cuda.is_available():
        logger.warning("CUDA nem elérhető, a predikció CPU-n fut.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    # Modell betöltése
    model_file = Path(model_path)
    if not model_file.exists():
        logger.error(f"Modell fájl nem található: {model_path}")
        logger.error("Kérlek, futtasd először a 02-training.py szkriptet!")
        return
    
    try:
        model = CoralModel(MODEL_NAME, num_classes=NUM_CLASSES, extra_feat_dim=len(FEATURE_COLS))
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()
        logger.info("✓ Modell sikeresen betöltve.\n")
    except Exception as e:
        logger.error(f"Hiba a modell betöltése közben: {e}")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Predikciók futtatása minden holdout mintán
    predictions = []
    ground_truths = []
    
    for idx, row in df_holdout.iterrows():
        text = row['paragraph_text']
        true_label = row['label_int']
        
        logger.info("#" * 80)
        logger.info(f"HOLDOUT PÉLDA #{idx + 1}")
        logger.info("#" * 80)
        logger.info(f"Task ID: {row.get('task_id', 'N/A')}")
        logger.info(f"Valós címke: {true_label}")
        logger.info(f"Szöveg hossza: {len(text)} karakter")
        logger.info(f"Szöveg: '{text[:150]}{'...' if len(text) > 150 else ''}'") 
        
        # Jellemzők kinyerése (az adathalmazból)
        features = torch.tensor([row[col] for col in FEATURE_COLS], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Tokenizálás
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_LEN)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Forward pass
            probs, _ = model(input_ids, attention_mask, extra_feats=features)
            
            # CORAL prediction
            B = probs.size(0)
            ones = torch.ones(B, 1, device=device)
            zeros = torch.zeros(B, 1, device=device)
            p_gt = torch.cat([ones, probs, zeros], dim=1)
            p_exact = p_gt[:, :-1] - p_gt[:, 1:]
            label_values = torch.arange(1, NUM_CLASSES + 1, device=device).float()
            exp_val = torch.sum(p_exact * label_values, dim=1)
            predicted_label = torch.clamp(torch.round(exp_val), 1, NUM_CLASSES).long().item()
        
        predictions.append(predicted_label)
        ground_truths.append(true_label)
        
        # Eredmény kiírása
        logger.info("\n" + "-" * 80)
        logger.info("EREDMÉNY:")
        logger.info(f"  Valós címke:      {true_label} / {NUM_CLASSES}")
        logger.info(f"  Prediktált címke: {predicted_label} / {NUM_CLASSES}")
        
        error = abs(predicted_label - true_label)
        if error == 0:
            logger.info(f"  ✓ PONTOS predikció (MAE: 0.0)")
        else:
            logger.info(f"  ✗ Eltérés: {error} szint (MAE: {error:.1f})")
        
        logger.info("-" * 80 + "\n")
    
    # Összesített eredmények
    logger.info("\n" + "=" * 80)
    logger.info("HOLDOUT SET ÖSSZESÍTETT EREDMÉNYEK")
    logger.info("=" * 80)
    
    mae = sum(abs(p - g) for p, g in zip(predictions, ground_truths)) / len(predictions)
    accuracy = sum(p == g for p, g in zip(predictions, ground_truths)) / len(predictions) * 100
    
    logger.info(f"Holdout minták száma: {len(predictions)}")
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Pontos találatok: {sum(p == g for p, g in zip(predictions, ground_truths))} / {len(predictions)}")
    logger.info(f"Accuracy: {accuracy:.1f}%")
    
    logger.info("\nRészletes eredmények:")
    for i, (pred, true) in enumerate(zip(predictions, ground_truths), 1):
        logger.info(f"  Példa {i}: Predikció={pred}, Valós={true}, Hiba={abs(pred-true)}")
    
    logger.info("=" * 80 + "\n")
    
    return predictions, ground_truths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predikció holdout (unseen) adatokon.")
    parser.add_argument(
        "--holdout_path",
        type=str,
        default=INFERENCE_HOLDOUT_PATH,
        help="A holdout adat CSV fájl elérési útja."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(Path(MODELS_DIR) / "coral_fold1_best.bin"),
        help="A betanított modell .bin fájljának elérési útja."
    )
    args = parser.parse_args()
    
    run_inference_on_holdout(args.holdout_path, args.model_path)
