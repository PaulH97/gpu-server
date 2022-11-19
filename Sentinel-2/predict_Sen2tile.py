# import rasterio
# from tools_preprocess import resampleRaster, patchifyRasterAsArray, savePatchesPredict, calculateIndizesSen2

import os
import numpy as np
from glob import glob

import tensorflow as tf 
from tools_model import load_img_as_array, predictPatches, dice_coef
from datagen import CustomImageGeneratorPrediction
from glob import glob

sentinel2_folder = "/codede/Sentinel-2/MSI/L3-WASP/2021/06/01/S2_L3_WASP_202106_32UNA"
sentinel_paths = glob("{}/*.tif".format(sentinel2_folder))
model_path = "/home/hoehn/data/output/Sentinel-2/models/Ad_bc_50_idx"
output_folder = "/home/hoehn/data/prediction"
patch_size = 128

# bands_patches = {}

# for idx, band in enumerate(sentinel_paths):

#     band_name = os.path.basename(band).split("_")[-1].split(".")[0]
#     print("Start patching with band: ", band_name)
#     raster = rasterio.open(band)
    
#     if raster.transform[0] != 10:  
#         raster = resampleRaster(band, 10)
#         r_array = raster.ReadAsArray()
#         r_array = np.expand_dims(r_array, axis=0)
#     else:
#         r_array = raster.read()[:,:10980,:10980]
    
#     r_array = np.moveaxis(r_array, 0, -1)
#     r_array = np.nan_to_num(r_array)
    
#     a,b = 0,1
#     c,d = np.percentile(r_array, [0.1, 99.9])
#     r_array_norm = (b-a)*((r_array-c)/(d-c))+a
#     r_array_norm[r_array_norm > 1] = 1
#     r_array_norm[r_array_norm < 0] = 0 

#     bands_patches[band_name] = patchifyRasterAsArray(r_array_norm, patch_size)

# bands_patches = calculateIndizesSen2(bands_patches)

# patches_path = savePatchesPredict(bands_patches, output_folder)

model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'dice_coef': dice_coef})
patches_path = glob(r"{}/full_img/32_UNA/*.tif".format(output_folder))
patches_path = sorted(patches_path, key = lambda x: int(x.split("_")[-1].split(".")[0]))

patch_array = load_img_as_array(patches_path[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

predict_datagen = CustomImageGeneratorPrediction(patches_path, patch_xy, b_count)

predictPatches(model, predict_datagen, sentinel_paths[4], output_folder)
