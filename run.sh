#!/bin/bash

# Ez a szkript futtatja a teljes tanítási folyamatot.
# 1. Telepíti a függőségeket (ha szükséges, bár a Docker ezt kezeli)
# 2. Futtatja a fő Python szkriptet a tanításhoz.

echo "Starting the training process..."

# A fő szkript futtatása
# A `main.py` fogja vezérelni az adat-előfeldolgozást, tanítást és kiértékelést.
python3 src/main.py

echo "Training process finished."
