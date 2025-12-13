# Legal Text Readability Prediction

This project implements a deep learning model that predicts the readability of Hungarian legal text paragraphs on a 1-5 scale (1=very difficult, 5=very easy).

## Project Information
- **Topic**: Legal Text Decoder
- **Student Name**: Gál Benjamin
- **Target +1 Grade**: No

## Solution Description

The model predicts readability levels of Hungarian legal documents (Gránit Bank Terms & Conditions excerpts) using an ordinal regression approach. The solution combines:

1. **Feature Engineering**: 23 handcrafted features extracted via spaCy Hungarian NLP pipeline
   - Readability metrics (Flesch, Gunning Fog, SMOG)
   - Lexical complexity (TTR, word length, complex word ratio)
   - Legal domain features (legal terms, abbreviations frequency)
   - Syntactic depth (dependency tree statistics)

2. **Baseline Model**: LogisticAT ordinal regression (mord library) using only the 23 features

3. **Deep Learning Model**: Hungarian BERT (`SZTAKI-HLT/hubert-base-cc`) with CORAL head
   - Hybrid architecture: Transformer embeddings + handcrafted features
   - CORAL (Cumulative Ordinal Regression) ensures ordinal consistency
   - 5-fold cross-validation with early stopping
   - Metrics: MAE (Mean Absolute Error), QWK (Quadratic Weighted Kappa)

The pipeline runs fully automated in Docker, logging all training details to stdout.

## Project Structure

The repository does not include data files, trained models, or logs - these are generated during runtime and mounted via Docker volumes.

```
dl_project_legal_text_decoder/
├── src/
│   ├── 01-data-preprocessing.py    # Data loading, EDA, feature extraction
│   ├── 02-training.py               # Model training with 5-fold CV
│   ├── 03-evaluation.py             # Test set evaluation
│   ├── 04-inference.py              # CLI prediction tool
│   ├── config.py                    # Configuration and hyperparameters
│   ├── model.py                     # PyTorch model definitions
│   └── utils.py                     # Logger utilities
├── notebooks/
│   └── notebook_best.ipynb          # Experimental notebook
├── data/                            # Data directory (mounted from host, gitignored except .gitkeep)
├── log/                             # Training logs (gitignored, .gitkeep tracked)
├── models/                          # Saved models (gitignored, .gitkeep tracked)
├── Dockerfile                       # Docker image definition
├── requirements.txt                 # Python dependencies (pinned versions)
├── run.sh                           # Orchestration script (01→02→03→04)
└── README.md                        # This documentation
```

## Data Preparation

The project includes an empty `data/` directory that will be mounted during Docker execution.

**Automatic Download (Recommended)**: 
- The pipeline automatically downloads `granit_bank_cimkezes.json` from Google Drive if not present
- No manual download needed - just run the Docker container!


## Docker Instructions

### Build
Build the Docker image from the repository root:

```bash
docker build -t legal-text-decoder .
```

### Ruthe project's `data/` directory to persist downloaded files, processed data, and trained models:

**Windows (PowerShell):**
```powershell
docker run --rm -v "${PWD}\data:/app/data" legal-text-decoder > log/run.log 2>&1
```

**Linux/Mac:**
```bash
docker run --rm -v "$(pwd)/data:/app/data" legal-text-decoder > log/run.log 2>&1
```

**What gets saved in `data/`:**
- `granit_bank_cimkezes.json` - Annotation file (auto-downloaded if missing)
- `processed_data.csv` - Training data with extracted features
- `inference_holdout.csv` - Holdout examples for inference
- `models/` - Trained CORAL models (`coral_fold1_best.bin` ... `coral_fold5_best.bin`)

**Note**: All directories are automatically created. The complete execution log is saved to `log/run.log`.
**Note**: The `log/`, `models/`, and `data/` directories are automatically created if they don't exist - no manual setup required!

The complete execution log will be saved to `log/run.log`.

## Pipeline Stages

The `run.sh` script executes the following stages sequentially:

1. **01-data-preprocessing.py**: 
   - Loads JSON annotations
   - **Randomly selects 2 examples as holdout set (unseen data)**
   - Saves holdout examples to `data/inference_holdout.csv`
   - Performs exploratory data analysis (EDA) on training data
   - Extracts 23 linguistic features using spaCy
   - Saves processed training data to `data/processed_data.csv`

2. **02-training.py**:
   - Trains CORAL model with 5-fold cross-validation **on training data only**
   - Logs model architecture, training metrics (loss, MAE, QWK)
   - Saves best model per fold
   - Generates confusion matrices

3. **03-evaluation.py**:
   - Evaluates baseline and transformer models on test fold
   - Reports final MAE and QWK scores

4. **04-inference.py**:
   - **Runs predictions on 2 holdout examples (truly unseen data)**
   - These examples were excluded from training/validation
   - Compares predictions with ground truth labels
   - Reports MAE and accuracy on holdout set

## Logging Requirements

All required logging components are implemented and output to `log/run.log`:

1. **Configuration**: Model name, hyperparameters (batch size, learning rate, epochs, etc.)
2. **Data Analysis**: Record counts, label distribution, text length statistics, class balance, outlier detection
3. **Model Architecture**: Parameter counts (backbone + head), layer details
4. **Training Metrics**: Per-epoch train/validation loss, MAE, QWK
5. **Validation Results**: Best MAE per fold, confusion matrices
6. **Final Evaluation**: Test set performance for baseline and transformer models
7. **Inference Example**: Model loading and prediction demonstration
