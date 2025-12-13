"""
Adat-előfeldolgozási funkciók.
"""
import json
import re
import math
import pandas as pd
from pathlib import Path
import spacy
from tqdm import tqdm
import numpy as np
import urllib.request
import urllib.error

# Jogi terminusok és rövidítések a configból (vagy ide is helyezhetők)
LEGAL_TERMS = [
    "szerződés","feltétel","jog","kötelezettség","felelősség","kártérítés","hatály","rendelkezés",
    "törvény","rendelet","bíróság","per","felmondás","biztosítás","ügyfél","jogosult","kötelezett",
    "igény","teljesítés","megszűnés","érvényesség","jogviszony","követelés","eljárás","határozat",
    "jogorvoslat","kikötés","megállapodás","szerződő","felek","szolgáltatás","ellenszolgáltatás",
    "jogosultság","kötelezettségvállalás","közjegyző","polgári","meghatalmazás","meghatalmazott"
]
LEGAL_ABBREVIATIONS = ["ptk","kft","zrt","bt","áfa","ászf","gvh","mkeh","mnb","pkkr"]
VOWELS = "aáeéiíoóöőuúüű"

# spaCy modell betöltése
try:
    # A Docker image-ben a modell a /app/hu_core_news_md utvonalon lesz
    nlp = spacy.load("hu_core_news_md")
    print("OK spaCy modell (hu_core_news_md) betoltve.")
except OSError:
    print("! hu_core_news_md nem talalhato. Fallback: blank magyar modell.")
    nlp = spacy.blank('hu')
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')


def download_from_google_drive(file_id: str, destination: str, logger) -> bool:
    """
    Letölt egy fájlt Google Drive-ról a megadott file ID alapján.
    
    Args:
        file_id: Google Drive file ID (a linkből)
        destination: Hova mentse a fájlt
        logger: Logger objektum
    
    Returns:
        True ha sikeres, False ha nem
    """
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        logger.info(f"Letoltes Google Drive-rol: {file_id}")
        
        # Ellenőrzés: szükséges-e confirmation token (nagy fájlok esetén)
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        
        # Először próbáljuk meg közvetlenül letölteni
        content = response.read()
        
        # Ha a válasz HTML és tartalmazza a "virus scan warning"-et, akkor confirmation kell
        if b'<!DOCTYPE html>' in content[:100]:
            # Keressuk meg a confirmation tokent
            logger.warning("Nagy fajl - confirmation token szukseges")
            # Egyszerűsített: próbáljuk confirm=t paraméterrel
            url_with_confirm = f"{url}&confirm=t"
            response = urllib.request.urlopen(url_with_confirm)
            content = response.read()
        
        # Mentés
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        with open(destination, 'wb') as f:
            f.write(content)
        
        logger.info(f"Sikeres letoltes: {destination}")
        return True
        
    except urllib.error.URLError as e:
        logger.error(f"HIBA a letoltes soran: {e}")
        return False
    except Exception as e:
        logger.error(f"Varatlan hiba a letoltes soran: {e}")
        return False

def load_annotation_json(path: str, logger) -> pd.DataFrame:
    """
    Betölti a címkézési adatokat egy JSON fájlból és DataFrame-mé alakítja.
    Ha a fájl nem létezik, megpróbálja letölteni Google Drive-ról.
    """
    p = Path(path)
    
    # Ha nem letezik, probaljuk meg letolteni
    if not p.exists():
        logger.warning(f"Adatfajl nem talalhato: {path}")
        logger.info("Megprobalom letolteni Google Drive-rol...")
        
        # Google Drive file ID a megosztott linkből
        GDRIVE_FILE_ID = "19UlAsuzprmhTl_I5Z_58d7AAIw3eJX_l"
        
        if download_from_google_drive(GDRIVE_FILE_ID, path, logger):
            logger.info("Sikeres letoltes! Folytatom a feldolgozast...")
        else:
            logger.error(f"Nem sikerult letolteni a fajlt. Adatfajl nem talalhato: {path}")
            return pd.DataFrame(columns=['task_id','paragraph_text','label_int','label_text'])
    raw = p.read_text(encoding='utf-8').strip()
    if not raw:
        logger.warning("A JSON fajl ures.")
        return pd.DataFrame(columns=['task_id','paragraph_text','label_int','label_text'])
    if not raw.startswith('['):
        raw = f'[{raw}]'
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f'JSON decode hiba: {e}')
        return pd.DataFrame(columns=['task_id','paragraph_text','label_int','label_text'])
    
    rows = []
    for task in data:
        text = task.get('data', {}).get('text', '').strip()
        ann_list = task.get('annotations', [])
        if not ann_list or not ann_list[0].get('result'):
            continue
        choice = ann_list[0]['result'][0].get('value', {}).get('choices', [None])[0]
        if not choice:
            continue
        m = re.match(r'(\d)', str(choice))
        if not m:
            continue
        rows.append({
            'task_id': task.get('id'),
            'paragraph_text': text,
            'label_int': int(m.group(1)),
            'label_text': choice
        })
    logger.info(f"Sikeresen betoltve {len(rows)} cimkezett adatpont.")
    return pd.DataFrame(rows)

def count_syllables_hu(word):
    count = 0
    in_grp = False
    for ch in word.lower():
        if ch in VOWELS:
            if not in_grp:
                count += 1
                in_grp = True
        else:
            in_grp = False
    return max(1, count)

def extract_features(text: str) -> dict:
    """Kinyeri a szöveges jellemzőket egy adott szövegből."""
    words = re.findall(r'\w+', text.lower())
    word_count = len(words)
    char_count = len(text)
    sentence_count = max(1, len(re.findall(r'[.!?]', text)))
    syllable_count = sum(count_syllables_hu(w) for w in words) if words else 0
    avg_words_per_sentence = word_count / sentence_count if sentence_count else 0
    avg_syllables_per_word = syllable_count / word_count if word_count else 0
    
    flesch_score_hu = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word if word_count else 0
    
    complex_words = sum(1 for w in words if count_syllables_hu(w) >= 3) if words else 0
    complex_word_ratio = complex_words / word_count if word_count else 0
    gunning_fog = 0.4 * (avg_words_per_sentence + 100 * complex_word_ratio) if word_count else 0
    
    smog_index = 1.0430 * math.sqrt(complex_words * (30 / sentence_count)) + 3.1291 if sentence_count and complex_words else 0
    
    unique_words = len(set(words)) if words else 0
    ttr = unique_words / word_count if word_count else 0
    
    legal_term_ratio = sum(1 for w in words if w in LEGAL_TERMS) / word_count if word_count else 0
    legal_abbr_ratio = sum(1 for w in words if w in LEGAL_ABBREVIATIONS) / word_count if word_count else 0
    
    long_word_ratio = sum(1 for w in words if len(w) > 12) / word_count if word_count else 0
    avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0
    
    comma_ratio = text.count(',') / word_count if word_count else 0
    parenthesis_ratio = (text.count('(') + text.count(')')) / sentence_count if sentence_count else 0
    uppercase_ratio = sum(1 for w in re.findall(r'\b[A-ZÁÉÍÓÖŐÚÜŰ]+\b', text)) / word_count if word_count else 0
    
    doc = nlp(text)
    num_entities = len(doc.ents)
    pos_counts = doc.count_by(spacy.attrs.POS)
    pos_noun_ratio = pos_counts.get(spacy.symbols.NOUN, 0) / word_count if word_count else 0
    pos_verb_ratio = pos_counts.get(spacy.symbols.VERB, 0) / word_count if word_count else 0
    pos_adj_ratio = pos_counts.get(spacy.symbols.ADJ, 0) / word_count if word_count else 0
    pos_adv_ratio = pos_counts.get(spacy.symbols.ADV, 0) / word_count if word_count else 0
    
    depths = [sum(1 for _ in token.ancestors) for token in doc]
    avg_dep_depth = float(np.mean(depths)) if depths else 0
    max_dep_depth = float(max(depths)) if depths else 0
    
    return {
        'char_count': char_count, 'word_count': word_count, 'sentence_count': sentence_count,
        'syllable_count': syllable_count, 'avg_word_length': avg_word_length,
        'flesch_score_hu': flesch_score_hu, 'gunning_fog': gunning_fog, 'smog_index': smog_index,
        'ttr': ttr, 'complex_word_ratio': complex_word_ratio, 'legal_term_ratio': legal_term_ratio,
        'legal_abbr_ratio': legal_abbr_ratio, 'long_word_ratio': long_word_ratio,
        'comma_ratio': comma_ratio, 'parenthesis_ratio': parenthesis_ratio,
        'uppercase_ratio': uppercase_ratio, 'num_entities': num_entities,
        'pos_noun_ratio': pos_noun_ratio, 'pos_verb_ratio': pos_verb_ratio,
        'pos_adj_ratio': pos_adj_ratio, 'pos_adv_ratio': pos_adv_ratio,
        'avg_dep_depth': avg_dep_depth, 'max_dep_depth': max_dep_depth
    }

def analyze_data(df: pd.DataFrame, logger) -> None:
    """
    Részletes feltáró adatelemzés (EDA) és naplózás.
    """
    logger.info("\n" + "=" * 80)
    logger.info("ADATELEMZÉS (Exploratory Data Analysis)")
    logger.info("=" * 80)
    
    # 1. Alapstatisztikak
    logger.info(f"\n1. ALAPSTATISZTIKAK")
    logger.info(f"   Teljes rekordszam: {len(df)}")
    logger.info(f"   Oszlopok szama: {len(df.columns)}")
    logger.info(f"   Oszlopok: {list(df.columns)}")
    
    # Hianyzó ertekek ellenorzese
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        logger.warning(f"   ! Hianyzó ertekek osszesen: {total_missing}")
        for col, count in missing[missing > 0].items():
            logger.warning(f"      - {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        logger.info(f"   OK Nincs hianyzó ertek az adatokban")
    
    # 2. Cimke eloszlas reszletesen
    logger.info(f"\n2. CIMKE ELOSZLAS")
    label_counts = df['label_int'].value_counts().sort_index()
    logger.info(f"   Erthetosegi szintek (1=nagyon nehez, 5=nagyon konnyu):")
    for label, count in label_counts.items():
        percentage = count/len(df)*100
        bar = '#' * int(percentage / 2)  # Visual bar chart
        logger.info(f"   Szint {label}: {count:3d} ({percentage:5.1f}%) {bar}")
    
    # Osztály-egyensúly ellenőrzése
    min_class = label_counts.min()
    max_class = label_counts.max()
    imbalance_ratio = max_class / min_class
    logger.info(f"\n   Osztaly-egyensuly elemzes:")
    logger.info(f"   - Legkisebb osztaly: {min_class} pelda")
    logger.info(f"   - Legnagyobb osztaly: {max_class} pelda")
    logger.info(f"   - Egyensulytalansagi arany: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        logger.warning(f"   ! Jelentes osztaly-egyensulytalansag ({imbalance_ratio:.1f}x)")
    else:
        logger.info(f"   OK Elfogadhato osztaly-egyensuly")
    
    # 3. Szoveg hossz statisztikak
    logger.info(f"\n3. SZOVEG HOSSZ STATISZTIKAK")
    df['_text_length'] = df['paragraph_text'].str.len()
    df['_word_count'] = df['paragraph_text'].str.split().str.len()
    
    logger.info(f"   Karakter hossz:")
    logger.info(f"   - Minimum: {df['_text_length'].min()} karakter")
    logger.info(f"   - Maximum: {df['_text_length'].max()} karakter")
    logger.info(f"   - Atlag: {df['_text_length'].mean():.1f} karakter")
    logger.info(f"   - Median: {df['_text_length'].median():.1f} karakter")
    logger.info(f"   - Std: {df['_text_length'].std():.1f}")
    
    logger.info(f"\n   Szoszam:")
    logger.info(f"   - Minimum: {df['_word_count'].min()} szo")
    logger.info(f"   - Maximum: {df['_word_count'].max()} szo")
    logger.info(f"   - Atlag: {df['_word_count'].mean():.1f} szo")
    logger.info(f"   - Median: {df['_word_count'].median():.1f} szo")
    logger.info(f"   - Std: {df['_word_count'].std():.1f}")
    
    # 4. Cimke-hossz korrelacio
    logger.info(f"\n4. CIMKE-SZOVEGHOSSZ KORRELACIO")
    avg_length_by_label = df.groupby('label_int')['_word_count'].agg(['mean', 'std'])
    logger.info(f"   Atlagos szoszam nehezsegi szintenkent:")
    for label in sorted(df['label_int'].unique()):
        mean_words = avg_length_by_label.loc[label, 'mean']
        std_words = avg_length_by_label.loc[label, 'std']
        logger.info(f"   Szint {label}: {mean_words:6.1f} +/- {std_words:5.1f} szo")
    
    # Pearson korrelacio szamitas
    correlation = df['label_int'].corr(df['_word_count'])
    logger.info(f"\n   Pearson korrelacio (cimke ~ szoszam): {correlation:.3f}")
    if abs(correlation) < 0.1:
        logger.info(f"   -> Gyenge korrelacio: hossz onmagaban nem meghatarozo")
    elif abs(correlation) < 0.3:
        logger.info(f"   -> Kozepes korrelacio: hossz reszben relevans")
    else:
        logger.info(f"   -> Eros korrelacio: hossz jelentes tenyezo")
    
    # 5. Jogi kifejezesek gyakorisaga
    logger.info(f"\n5. JOGI DOMAIN JELLEMZOK")
    df['_has_legal_term'] = df['paragraph_text'].str.lower().str.contains('|'.join(LEGAL_TERMS), regex=True)
    df['_has_legal_abbr'] = df['paragraph_text'].str.lower().str.contains('|'.join(LEGAL_ABBREVIATIONS), regex=True)
    
    legal_term_count = df['_has_legal_term'].sum()
    legal_abbr_count = df['_has_legal_abbr'].sum()
    
    logger.info(f"   Jogi kifejezest tartalmazo bekezdesek: {legal_term_count} ({legal_term_count/len(df)*100:.1f}%)")
    logger.info(f"   Jogi roviditest tartalmazo bekezdesek: {legal_abbr_count} ({legal_abbr_count/len(df)*100:.1f}%)")
    
    
    # 6. Outlier detekció
    logger.info(f"\n6. OUTLIER DETEKCIÓ (Szoveghossz alapjan)")
    Q1 = df['_word_count'].quantile(0.25)
    Q3 = df['_word_count'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['_word_count'] < lower_bound) | (df['_word_count'] > upper_bound)]
    logger.info(f"   IQR modszer (1.5 x IQR):")
    logger.info(f"   - Q1 (25%): {Q1:.1f} szo")
    logger.info(f"   - Q3 (75%): {Q3:.1f} szo")
    logger.info(f"   - IQR: {IQR:.1f}")
    logger.info(f"   - Also korlat: {lower_bound:.1f} szo")
    logger.info(f"   - Felso korlat: {upper_bound:.1f} szo")
    logger.info(f"   - Outlier-ek szama: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    if len(outliers) > 0:
        logger.info(f"   Outlier peldak:")
        for idx, row in outliers.head(3).iterrows():
            logger.info(f"     - {row['_word_count']} szo (szint {row['label_int']}): '{row['paragraph_text'][:80]}...'")
    
    # 7. Szokincs statisztikak
    logger.info(f"\n7. SZOKINCS MERET")
    all_words = ' '.join(df['paragraph_text'].str.lower()).split()
    unique_words = set(all_words)
    logger.info(f"   Osszes szo: {len(all_words):,}")
    logger.info(f"   Egyedi szavak: {len(unique_words):,}")
    logger.info(f"   Szokincs gazdagsag (Type-Token Ratio): {len(unique_words)/len(all_words):.4f}")
    
    # Leggyakoribb szavak
    from collections import Counter
    word_freq = Counter(all_words)
    logger.info(f"\n   10 leggyakoribb szo:")
    for word, count in word_freq.most_common(10):
        logger.info(f"     - '{word}': {count} elofordulas")
    
    # Tisztitas: temporary oszlopok eltavolitasa
    df.drop(['_text_length', '_word_count', '_has_legal_term', '_has_legal_abbr'], axis=1, inplace=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("ADATELEMZES BEFEJEZVE")
    logger.info("=" * 80 + "\n")

def get_processed_data(annotation_path: str, output_path: str, logger) -> None:
    """
    Betölti a címkéket, kinyeri a jellemzőket és elmenti őket egyetlen CSV fájlba.
    ÚJ: 2 példát elkülönít inference teszteléshez (holdout set).
    """
    df_labels = load_annotation_json(annotation_path, logger)
    if df_labels.empty:
        logger.error("Nincsenek betoltheto adatok, a folyamat leall.")
        return

    logger.info(f"Betoltott cimkek szama: {len(df_labels)}")
    
    # --- UJ: Holdout minta kivalasztasa inference teszteleshez ---
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE HOLDOUT MINTA KIVALASZTASA")
    logger.info("=" * 80)
    
    # Fix seed a reprodukálhatóságért
    np.random.seed(42)
    
    # Stratified sampling: vegyen 1-1 példát különböző nehézségi szintekből
    # hogy lássuk a modell teljesítményét különböző nehézségeken
    holdout_indices = []
    available_labels = df_labels['label_int'].unique()
    
    # Válasszuk ki a 2. és 4. nehézségi szintből (középső tartomány)
    target_labels = [2, 4] if 2 in available_labels and 4 in available_labels else available_labels[:2]
    
    for label in target_labels[:2]:  # Maximum 2 példa
        label_indices = df_labels[df_labels['label_int'] == label].index.tolist()
        if label_indices:
            selected_idx = np.random.choice(label_indices, size=1)[0]
            holdout_indices.append(selected_idx)
    
    # Holdout és training adatok szétválasztása
    df_holdout = df_labels.loc[holdout_indices].copy()
    df_training = df_labels.drop(holdout_indices).reset_index(drop=True)
    
    logger.info(f"Holdout mintak szama: {len(df_holdout)}")
    logger.info(f"Training mintak szama: {len(df_training)}")
    logger.info("\nKivalasztott holdout peldak:")
    for idx, row in df_holdout.iterrows():
        logger.info(f"  - Task ID {row['task_id']}, Cimke: {row['label_int']}, Szoveg: '{row['paragraph_text'][:80]}...'")
    logger.info("=" * 80 + "\n")
    
    # --- Részletes adat analízis (csak training adaton) ---
    analyze_data(df_training, logger)
    
    # --- Folytatas a jellemzokinyeressel (TRAINING adat) ---
    logger.info("\n--- Jellemzokinyeres (Training adat) ---")
    feature_rows_train = [extract_features(t) for t in tqdm(df_training['paragraph_text'], desc='Training Feature Extraction')]
    df_feat_train = pd.DataFrame(feature_rows_train)
    df_processed_train = pd.concat([df_training.reset_index(drop=True), df_feat_train], axis=1)
    
    # --- Fold assignment hozzaadasa (K-Fold CV-hez) ---
    from sklearn.model_selection import StratifiedKFold
    logger.info("\n--- K-Fold assignment (5-fold stratified split) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_processed_train['fold'] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_processed_train, df_processed_train['label_int'])):
        df_processed_train.loc[val_idx, 'fold'] = fold_idx
    logger.info(f"Fold eloszlas:\n{df_processed_train['fold'].value_counts().sort_index()}")
    
    # --- Jellemzokinyeres (HOLDOUT adat) ---
    logger.info("\n--- Jellemzokinyeres (Holdout/Inference adat) ---")
    feature_rows_holdout = [extract_features(t) for t in tqdm(df_holdout['paragraph_text'], desc='Holdout Feature Extraction')]
    df_feat_holdout = pd.DataFrame(feature_rows_holdout)
    df_processed_holdout = pd.concat([df_holdout.reset_index(drop=True), df_feat_holdout], axis=1)
    
    # Elmentjük a feldolgozott adatokat
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training adat mentese
    df_processed_train.to_csv(output_path, index=False)
    logger.info(f"Training adat elmentve: {output_path} ({len(df_processed_train)} sor)")
    
    # Holdout adat mentese kulon fajlba
    holdout_path = output_path.replace('processed_data.csv', 'inference_holdout.csv')
    df_processed_holdout.to_csv(holdout_path, index=False)
    logger.info(f"Inference holdout adat elmentve: {holdout_path} ({len(df_processed_holdout)} sor)")
    
    logger.info(f"\nAz adat-elofeldolgozas befejezodott.")

from utils import get_logger
from config import (
    ANNOTATION_PATH, PROCESSED_DATA_PATH, SEED, DEVICE, MODEL_NAME,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, KFOLDS, PATIENCE, NUM_CLASSES, MAX_LEN
)

if __name__ == '__main__':
    logger = get_logger(__name__)
    logger.info("=" * 80)
    logger.info("LEGAL TEXT DECODER - Adatfeldolgozás és Modell Tanítási Pipeline")
    logger.info("=" * 80)
    
    logger.info("\n--- KONFIGURACIÓS PARAMETEREK ---")
    logger.info(f"Random Seed: {SEED}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Transformer Modell: {MODEL_NAME}")
    logger.info(f"Osztalyok szama: {NUM_CLASSES} (1-5 skala)")
    logger.info(f"Max tokenizalasi hossz: {MAX_LEN}")
    logger.info("\nTanitasi hiperparameterek:")
    logger.info(f"  - Batch Size: {BATCH_SIZE}")
    logger.info(f"  - Epochs: {EPOCHS}")
    logger.info(f"  - Learning Rate: {LEARNING_RATE}")
    logger.info(f"  - K-Fold CV: {KFOLDS} fold")
    logger.info(f"  - Early Stopping Patience: {PATIENCE} epoch")
    logger.info("\nUtvonalak:")
    logger.info(f"  - Annotacio: {ANNOTATION_PATH}")
    logger.info(f"  - Feldolgozott adat: {PROCESSED_DATA_PATH}")
    logger.info("-" * 80)
    
    logger.info("\n--- Adat-elofeldolgozasi Folyamat Inditasa ---")
    get_processed_data(
        annotation_path=ANNOTATION_PATH,
        output_path=PROCESSED_DATA_PATH,
        logger=logger
    )
    logger.info("--- Adat-elofeldolgozasi Folyamat Befejezodott ---")
