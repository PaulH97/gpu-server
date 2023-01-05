import rasterio
from patchify import patchify,unpatchify
import numpy as np
import os
from matplotlib import pyplot as plt
from keras import backend as K
import json
import tifffile as tiff
import random
from glob import glob

def load_img_as_array(path):
    # read img as array 
    img_array = rasterio.open(path).read()
    img_array = np.moveaxis(img_array, 0, -1)
    # img_array = np.nan_to_num(img_array)
    return img_array

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_distance_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    return -jaccard_distance_coef(y_true, y_pred)

def predictPatches(model, predict_datagen, raster_path, output_folder):

    prediction = model.predict(predict_datagen, verbose=1) 
    prediction = (prediction > 0.5).astype(np.uint8) 
    
    raster = rasterio.open(raster_path)
    patch_size = prediction.shape[1]
    x = int(raster.width/patch_size)
    y = int(raster.height/patch_size)
    patches_shape = (x, y, 1, patch_size, patch_size, 1)

    prediciton_reshape = np.reshape(prediction, patches_shape) 

    recon_predict = unpatchify(prediciton_reshape, (patch_size*x,patch_size*y,1))

    transform = raster.transform
    crs = raster.crs
    name = os.path.basename(raster_path).split("_")[3]
    predict_out = os.path.join(output_folder, name + "_prediction.tif")

    final = rasterio.open(predict_out, mode='w', driver='Gtiff',
                    width=recon_predict.shape[0], height=recon_predict.shape[1],
                    count=1,
                    crs=crs,
                    transform=transform,
                    dtype=rasterio.int16)

    final.write(recon_predict[:,:,0],1) 
    final.close()
    print("Reconstructed and predicted sentinel tile and saved in: ", predict_out)
    
    return recon_predict

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def createMetrics(file, metrics_dict, name_data):

    sum = {}
    append_new_line(file, f"--------------------- {name_data} ---------------------")
    for key, value in metrics_dict.items():
        append_new_line(file, f"--------------------- {str(key)} ---------------------")
        append_new_line(file, json.dumps(value))
        for k,v in value.items():
            if sum.get(k) is None:
                sum[k] = v
            else:
                sum[k] += v

    mean = {key: value/len(metrics_dict) for key, value in sum.items()}

    append_new_line(file, "--------------------- Mean metrics ---------------------")
    [append_new_line(file, f"{key}: {value}") for key, value in mean.items()]

def load_trainData(output_folder, idx):

    if idx:
        X_train = glob("{}/crops/idx/train/img/*.tif".format(output_folder))
        y_train = glob("{}/crops/idx/train/mask/*.tif".format(output_folder))
        X_test = glob("{}/crops/idx/test/img/*.tif".format(output_folder))
        y_test = glob("{}/crops/idx/test/mask/*.tif".format(output_folder))
    else:
        X_train = glob("{}/crops/no_idx/train/img/*.tif".format(output_folder))
        y_train = glob("{}/crops/no_idx/train/mask/*.tif".format(output_folder))
        X_test = glob("{}/crops/no_idx/test/img/*.tif".format(output_folder))
        y_test = glob("{}/crops/no_idx/test/mask/*.tif".format(output_folder))

    return X_train, X_test, y_train, y_test