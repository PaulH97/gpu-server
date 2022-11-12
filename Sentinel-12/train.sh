#!/bin/bash

echo "Change directory"
cd /home/hoehn/code

echo "Activate conda env for preprocess of training data"
conda activate gdal
python3 preprocess_training.py
conda deactivate gdal
conda deactivate 

echo "Activate virtual env and start training"
source /home/hoehn/code/env/bin/activate
python3 train.py 
deactivate

