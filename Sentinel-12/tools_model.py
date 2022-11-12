import rasterio
from patchify import patchify,unpatchify
import numpy as np
import os
from matplotlib import pyplot as plt
from keras import backend as K

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
    
    return recon_predict
