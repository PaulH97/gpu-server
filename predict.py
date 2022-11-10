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
if os.path.exists("config_prediction.yaml"):
    with open('config_prediction.yaml') as f:
        
        data = yaml.load(f, Loader=yaml.FullLoader)
        example_raster = glob("{}/*B2.tif".format(data['data_source']['Sentinel2']))[0]
        model_path = data['model']['path']
        output_folder = data["output_folder"]

model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'dice_coef': dice_coef})
patch_size = model.input_shape[1]

patches_path = glob(r"{}/Crops/img/*.tif".format(output_folder))
patches_path = sorted(patches_path, key = lambda x: int(x.split("_")[-1].split(".")[0]))

mask = load_img_as_array("/home/hoehn/data/prediction/solarParks_predict_S2_L3_WASP_202107_32UPV.tif")
mask = mask[:10880,:10880,:]
patch_array = load_img_as_array(patches_path[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

predict_datagen = CustomImageGeneratorPrediction(patches_path, patch_xy, b_count)

pred = predictPatches(model, predict_datagen, example_raster, output_folder)

# Get model metrics on test data
from sklearn.metrics import jaccard_score, f1_score, accuracy_score

flat_truth = np.concatenate(mask).flatten()
flat_pred = np.concatenate(pred).flatten()

# cm = confusion_matrix(flat_truth, flat_pred, normalize='true')

# fig, ax = plot_confusion_matrix(conf_mat=cm ,  figsize=(8,8))
# plt.title('Confusion matrix')
# plt.xticks(range(2), ['Normal','Solar'], fontsize=10)
# plt.yticks(range(2), ['Normal','Solar'], fontsize=10)
# plt.savefig(model_name + "cm.png") 

jaccard = jaccard_score(flat_truth, flat_pred)
f1 = f1_score(flat_truth, flat_pred,)
acc = accuracy_score(flat_truth, flat_pred)

print("Jaccard(%): ", jaccard*100)
print("F1-Score(%): ", f1*100)
print("Accuracy(%): ", acc*100)