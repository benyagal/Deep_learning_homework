# ÁSZF Érthetőség Predikciós Modell

Ez a projekt egy NLP modellt valósít meg, amely 1-től 5-ig terjedő skálán becsüli meg jogi szövegrészletek (jelen esetben a Gránit Bank ÁSZF-jének egy részlete) érthetőségét.

## Projekt Struktúra

A projekt a verziókezelőben nem tartalmazza az adatokat, a tanított modelleket és a logokat. Ezeket a futtatás során kell létrehozni és a konténerbe csatolni.

```
dl_project_legal_text_decoder/
├── notebooks/
│   └── notebook_best.ipynb         # Eredeti kísérletező notebook
├── src/
│   ├── __init__.py
│   ├── config.py                   # Konfigurációs fájl
│   ├── data_preprocessing.py       # Adatbetöltő és jellemző-kinyerő szkriptek
│   ├── model.py                    # PyTorch modell és adatkészlet definíciók
│   ├── train.py                    # Tanítási és kiértékelési logika
│   └── main.py                     # Fő belépési pont
├── .gitignore                      # Verziókezelésből kizárt fájlok
├── Dockerfile                      # Docker image definíció
├── requirements.txt                # Python függőségek
└── run.sh                          # Futtató szkript
```

## Projekt Részletek

### Projekt Információk
- **Téma**: Legal Text Decoder
- **Név**: [Ide írd a neved]
- **Cél a +1 jegy**: [Igen/Nem]

### Megoldás Leírása
A projekt célja egy mélytanuló modell létrehozása, amely képes megbecsülni magyar nyelvű jogi szövegrészletek érthetőségét egy 1-től 5-ig terjedő skálán. A megoldás egy `SZTAKI-HLT/hubert-base-cc` transzformer modellt használ alapul, amelyet egy ordinális klasszifikációra specializált **CORAL (Cumulative Ordinal Ranking and Regression)** kimeneti réteggel egészítünk ki. A modell a nyers szöveg mellett 23, a szöveg komplexitását, olvashatóságát és jogi jellegét leíró, kézzel készített jellemzőt is felhasznál a jobb teljesítmény érdekében. A tanítás 5-szörös keresztvalidációval történik, a kiértékelés fő metrikái a Mean Absolute Error (MAE) és a Quadratic Weighted Kappa (QWK). A teljes folyamat Docker konténerben fut a reprodukálhatóság biztosítása érdekében.

### Extra Pont Indoklása
[Ha a "+1 jegy" cél "Igen", itt indokold meg, hogy a munkád mely része (pl. innovatív modell architektúra, kiterjedt kísérletezés, kiemelkedő eredmény) érdemel extra pontot.]

### Adat-előkészítés
A modell tanításához a `granit_bank_cimkezes.json` fájl szükséges. Mivel ez nem része a repozitóriumnak, a futtatás előtt manuálisan kell beszerezni.

1.  **Hozzon létre egy `data` könyvtárat** a számítógépén egy tetszőleges, de megjegyezhető helyen (pl. `C:\Users\YourUser\Documents\dl_project_data`).
2.  **Töltse le a `granit_bank_cimkezes.json` fájlt** a következő SharePoint linkről:
    [BME VIK - Project Work - Adatfájl](https://bmeedu-my.sharepoint.com/shared?listurl=https%3A%2F%2Fbmeedu%2Dmy%2Esharepoint%2Ecom%2Fpersonal%2Fgyires%2Dtoth%5Fbalint%5Fvik%5Fbme%5Fhu%2FDocuments&id=%2Fpersonal%2Fgyires%2Dtoth%5Fbalint%5Fvik%5Fbme%5Fhu%2FDocuments%2FDokumentumok%2FVITMMA19%2F2025%2Fproject%2Dwork%2Flegaltextdecoder%2FQG1L1V&shareLink=1%2C1%2C1&ga=1)
3.  **Helyezze a letöltött fájlt** az imént létrehozott `data` könyvtárba.

A `src/data_preprocessing.py` szkript ezt a JSON fájlt olvassa be, kinyeri belőle a szövegeket és a címkéket, majd a `spaCy` és egyéni függvények segítségével legenerálja a szükséges 23 numerikus jellemzőt. Az adatok ezután készen állnak a modell által feldolgozható formátumra.

## Docker Instrukciók

A projekt konténerizált. Az alábbi lépésekkel építhető és futtatható.

### Build
Futtassa a következő parancsot a repozitórium gyökérkönyvtárában a Docker image felépítéséhez:

```bash
docker build -t legal-text-decoder .
```

### Futtatás
A futtatáshoz a `-v` kapcsolóval csatolni kell a helyi adatokat tartalmazó könyvtárat a konténer `/app/data` mappájához. A logok elmentéséhez irányítsa át a kimenetet egy fájlba.

**Fontos:** Cserélje le a `/path/to/your/local/data` részt a saját, "Adat-előkészítés" lépésben létrehozott `data` könyvtárának abszolút elérési útjára.

```bash
docker run --rm -v /path/to/your/local/data:/app/data legal-text-decoder > log/run.log 2>&1
```

Példa Windows alatt (PowerShell):
```powershell
# Hozd létre a log mappát, ha még nem létezik
if (-not (Test-Path -Path log)) { New-Item -ItemType Directory -Path log }
docker run --rm -v "C:\Users\YourUser\Documents\dl_project_data:/app/data" legal-text-decoder > log/run.log 2>&1
```

GPU használata esetén adja hozzá a `--gpus all` kapcsolót:
```bash
docker run --rm --gpus all -v /path/to/your/local/data:/app/data legal-text-decoder > log/run.log 2>&1
```
A parancs a teljes futás kimenetét a host gépen, a projekt gyökerében létrehozott `log/run.log` fájlba menti.

## Fájlstruktúra és Funkciók

- **`src/`**: A forráskódok.
  - **`main.py`**: A fő belépési pont. Elindítja az adatfeldolgozást, a baseline modell futtatását és a transzformer modell tanítását.
  - **`data_preprocessing.py`**: Betölti a JSON adatokat és kinyeri a numerikus jellemzőket.
  - **`model.py`**: Definiálja a `LegalDataset` és `CoralModel` osztályokat.
  - **`train.py`**: A teljes 5-szörös keresztvalidációs tanítási ciklust, kiértékelést és modellmentést vezérli.
  - **`config.py`**: Globális konfigurációs változók (hiperparaméterek, útvonalak).
  - **`utils.py`**: Segédfüggvények, jelenleg a központi logger beállításáért felel.
- **`notebooks/`**: Kísérletező Jupyter notebook.
- **`log/`**: A futtatási logok és a generált képek (tévesztési mátrixok) helye.
- **`models/`**: A tanítás során mentett modellfájlok helye.
- **`Dockerfile`**: A Docker image leírása.
- **`requirements.txt`**: Python függőségek listája.
- **`run.sh`**: A konténeren belüli futtatást vezérlő shell szkript.
- **`README.md`**: Ez a dokumentáció.

