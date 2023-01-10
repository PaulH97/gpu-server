import os
import yaml
import numpy as np
from glob import glob
import tensorflow as tf 
from tools_model import load_img_as_array, predictPatches, dice_coef
from datagen import CustomImageGeneratorPrediction
from glob import glob
from matplotlib import pyplot as plt

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)

        patch_size = data['model_parameter']['patch_size']
        indizes = data["indizes"]
        tile_folder = data['prediction']["data"]['Sentinel2']
        sentinel_paths = glob("{}/*.tif".format(tile_folder))
        model_path = data["prediction"]["model"]["path"]

tile_name = tile_folder.split("_")[-1]
output_folder = os.path.join("/home/hoehn/data/prediction", tile_name)

model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'dice_coef': dice_coef})
patches_path = glob(r"{}/full_img/*.tif".format(output_folder))
patches_path = sorted(patches_path, key = lambda x: int(x.split("_")[-1].split(".")[0]))

patch_array = load_img_as_array(patches_path[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

predict_datagen = CustomImageGeneratorPrediction(patches_path, patch_xy, b_count)

predictPatches(model, predict_datagen, sentinel_paths[4], output_folder)
