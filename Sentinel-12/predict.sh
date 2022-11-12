#!/bin/bash

echo "Activate conda env for preprocess of prediction"
conda activate gdal
python3 preprocess_prediction.py
conda deactivate gdal
conda deactivate 

echo "Make prediciton"
source /home/hoehn/code/env/bin/activate
python3 predict.py 
deactivate
