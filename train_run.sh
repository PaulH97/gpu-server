#!/bin/bash

# Schleife durch die Liste der Modellparameter
for i in {1..8}
do
  echo "Start with model ID: $i"
  if [$i -eq 7]
  then
    # Die aktuelle Nummer an das Python-Skript übergeben
    python train_bash.py --run_number=$i
    echo 
  fi
done