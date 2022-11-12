import rasterio
import tensorflow as tf 
import os
import numpy as np
from glob import glob
from datagen import CustomImageGeneratorPrediction
from tools_train_predict import dice_coef, predictPatches, load_img_as_array, jaccard_distance_coef
import yaml
from keras import backend as K

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:
        
        data = yaml.load(f, Loader=yaml.FullLoader)
        sentinel2 = data['prediction']['data']['Sentinel2']
        example_raster = glob("{}/*B2.tif".format(sentinel2))[0]
        model_path = data['prediction']['model']['path']
        output_folder = data["output_folder"]

model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'dice_coef': dice_coef})
patch_size = model.input_shape[1]

patches_path = glob(r"{}/prediction/crops/full_img/*.tif".format(output_folder))
patches_path = sorted(patches_path, key = lambda x: int(x.split("_")[-1].split(".")[0]))

patch_array = load_img_as_array(patches_path[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

predict_datagen = CustomImageGeneratorPrediction(patches_path, patch_xy, b_count)

prediction = predictPatches(model, predict_datagen, example_raster, output_folder)

# Get model metrics on test data
from sklearn.metrics import jaccard_score, f1_score, accuracy_score

tile_pred = sentinel2.split("_")[-1] # 32UPV
mask_path = glob("/home/hoehn/data/output/Sentinel-2/masks/*{}.tif".format(tile_pred))
mask = load_img_as_array(mask_path)
mask = mask[:10880,:10880,:]

flat_truth = np.concatenate(mask).flatten()
flat_pred = np.concatenate(prediction).flatten()

jaccard = jaccard_score(flat_truth, flat_pred)
f1 = f1_score(flat_truth, flat_pred,)
acc = accuracy_score(flat_truth, flat_pred)

print("Jaccard(%): ", jaccard*100)
print("F1-Score(%): ", f1*100)
print("Accuracy(%): ", acc*100)