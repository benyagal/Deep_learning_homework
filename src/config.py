"""
Központi konfigurációs fájl a projekthez.
Itt tároljuk a hiperparamétereket, fájlútvonalakat és egyéb beállításokat.
"""
import torch
from pathlib import Path

# -- Általános beállítások
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Adatfájlok útvonalai
# A konténeren belüli útvonalak. Az adatokat a `docker run -v` paranccsal kell csatolni.
DATA_DIR = "/app/data"
MODELS_DIR = "/app/models"
LOG_DIR = "/app/log"

# Automatikus mappa létrehozás (ha nem létezik)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
ANNOTATION_PATH = f"{DATA_DIR}/granit_bank_cimkezes.json"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed_data.csv"
INFERENCE_HOLDOUT_PATH = f"{DATA_DIR}/inference_holdout.csv"

# -- Adatkezelés
TEXT_COL = 'paragraph_text'
TARGET_COL = 'label_int'

# -- Modell és Tokenizer
MODEL_NAME = 'SZTAKI-HLT/hubert-base-cc'
BASELINE_MODEL_NAME = "log_at_baseline"
MAX_LEN = 256
NUM_CLASSES = 5

# -- Tanítási hiperparaméterek
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 1e-5
KFOLDS = 5
PATIENCE = 3

# -- Jellemzők (Features)
# A data_preprocessing.py által használt oszlopok listája
FEATURE_COLS = [
    'char_count', 'word_count', 'sentence_count', 'syllable_count',
    'avg_word_length', 'flesch_score_hu', 'gunning_fog', 'smog_index', 'ttr',
    'complex_word_ratio', 'legal_term_ratio', 'legal_abbr_ratio',
    'long_word_ratio', 'comma_ratio', 'parenthesis_ratio', 'uppercase_ratio',
    'num_entities', 'pos_noun_ratio', 'pos_verb_ratio', 'pos_adj_ratio',
    'pos_adv_ratio', 'avg_dep_depth', 'max_dep_depth'
]
