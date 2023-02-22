import os
import yaml
import numpy as np
from glob import glob
import tensorflow as tf 
from tools_model import load_img_as_array, predictPatches, dice_metric
from datagen import CustomImageGeneratorPrediction
from glob import glob
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, recall_score, precision_score
import rasterio

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        

        tile_folder = data['prediction']['Sentinel2']
        tile_id = tile_folder.split("_")[-1]
        sentinel_paths = glob("{}/*.tif".format(tile_folder))
        output_folder = data["output_folder"]
        prediction_folder = os.path.join(output_folder, "prediction", tile_id)
        model_name = data["model_parameter"]["name"]
        best_model = os.path.join(output_folder, "models", model_name, "best_model")
        tile_mask = os.path.join(prediction_folder, f"{tile_id}_mask.tif")

patch_size = model_name.split("_")[1][:3]
idx = model_name.split("_")[1][3:]

patches_path = glob("{}/{}/{}/full_img/*.tif".format(prediction_folder, patch_size, idx))
patches_path = sorted(patches_path, key = lambda x: int(x.split("_")[-1].split(".")[0]))

patch_array = load_img_as_array(patches_path[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

predict_datagen = CustomImageGeneratorPrediction(patches_path, patch_xy, b_count)

model = tf.keras.models.load_model(best_model, compile=False, custom_objects={'dice_metric': dice_metric})  

tile_pred = predictPatches(model, predict_datagen, sentinel_paths[4], prediction_folder)
xy = tile_pred.shape[0] 

tile_mask = load_img_as_array(tile_mask)

flat_truth = np.concatenate(tile_mask[:xy,:xy,:]).flatten()
flat_pred = np.concatenate(tile_pred).flatten()

# cm = confusion_matrix(flat_truth, flat_pred, normalize='true')

# fig, ax = plot_confusion_matrix(conf_mat=cm ,  figsize=(8,8))
# plt.title('Confusion matrix')
# plt.xticks(range(2), ['Normal','Solar'], fontsize=10)
# plt.yticks(range(2), ['Normal','Solar'], fontsize=10)
# plt.savefig(os.path.join(prediction_folder, model_name + "cm.png"))

# recall = recall_score(flat_truth, flat_pred)
precision = precision_score(flat_truth, flat_pred)
# jaccard = jaccard_score(flat_truth, flat_pred)
# f1 = f1_score(flat_truth, flat_pred,)
# acc = accuracy_score(flat_truth, flat_pred)

# print("Recall(%): ", recall*100)
print("Precision(%): ", precision*100)
# print("Jaccard(%): ", jaccard*100)
# print("F1-Score(%): ", f1*100)
# print("Accuracy(%): ", acc*100)