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
    # A Docker image-ben a modell a /app/hu_core_news_md útvonalon lesz
    nlp = spacy.load("/app/hu_core_news_md")
    print("✓ spaCy modell (hu_core_news_md) betöltve.")
except OSError:
    print("⚠ hu_core_news_md nem található. Fallback: blank magyar modell.")
    nlp = spacy.blank('hu')
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')


def load_annotation_json(path: str, logger) -> pd.DataFrame:
    """
    Betölti a címkézési adatokat egy JSON fájlból és DataFrame-mé alakítja.
    """
    p = Path(path)
    if not p.exists():
        logger.error(f"Adatfájl nem található: {path}")
        return pd.DataFrame(columns=['task_id','paragraph_text','label_int','label_text'])
    raw = p.read_text(encoding='utf-8').strip()
    if not raw:
        logger.warning("A JSON fájl üres.")
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
    logger.info(f"Sikeresen betöltve {len(rows)} címkézett adatpont.")
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

def get_processed_data(annotation_path: str, logger) -> pd.DataFrame:
    """
    Betölti a címkéket, kinyeri a jellemzőket és összefűzi őket egyetlen DataFrame-be.
    """
    df_labels = load_annotation_json(annotation_path, logger)
    if df_labels.empty:
        logger.error("Nincsenek betölthető adatok, a folyamat leáll.")
        return pd.DataFrame()
    
    logger.info(f"Betöltött címkék száma: {len(df_labels)}")
    logger.info("Címkék eloszlása:\n" + str(df_labels['label_int'].value_counts().sort_index()))
    
    feature_rows = [extract_features(t) for t in tqdm(df_labels['paragraph_text'], desc='Feature Extraction')]
    df_feat = pd.DataFrame(feature_rows)
    
    df_processed = pd.concat([df_labels.reset_index(drop=True), df_feat], axis=1)
    logger.info("Az adat-előfeldolgozás és jellemző-kinyerés befejeződött.")
    return df_processed
