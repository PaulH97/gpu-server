import os
from glob import glob
import yaml
import numpy as np
from tools_preprocess import *
import rasterio
from matplotlib import pyplot as plt
import shutil
import warnings
from sklearn.model_selection import train_test_split
import ast
import json

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# np.seterr(divide='ignore', invalid='ignore')
# warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
# GDAL_DISABLE_READDIR_ON_OPEN=True
# # -------------------- Load data -------------------------------
# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        # get data for preprocessing
        solar_path = data['shapefiles']['Solar']    
        patch_size = data['model_parameter']['patch_size']
        output_folder = data["output_folder"]
        indizes = data["indizes"]
        senTiles_file = data["satellite"]["tiles_path"]
        # data for prediction
        sentinel1_pred = data['prediction']['Sentinel1']
        sentinel2_pred = data['prediction']['Sentinel2']
        sentinel12_pred = [sentinel1_pred] + [sentinel2_pred]
        seed = data["seed"]
        # name of output directory -> see if Sentinel-1, Sentinel-2 or Sentinel-12
        dir_name = output_folder.split("/")[-1]
        tiles_file = data["satellite"]["tiles_path"]

# # # -------------------- Build folder structure ------------------------
# crop_name = "crop" + str(patch_size)
# crop_folder = os.path.join(output_folder, crop_name)

# if indizes:
#     crop_folder = os.path.join(crop_folder, "idx")
#     X_t_out, y_t_out, X_p_out, y_p_out, p_full_out = rebuildCropFolder(crop_folder)
# else:
#     crop_folder = os.path.join(crop_folder, "no_idx")
#     X_t_out, y_t_out, X_p_out, y_p_out, p_full_out = rebuildCropFolder(crop_folder)

# #Create following folder(full_img - img - mask) in predictrion folder
# # pred_cfolder = os.path.join(output_folder, "prediction", "crops")
# # full_img_out, imgPred_out, maskPred_out = rebuildPredFolder(pred_cfolder) 

# ##------------------- Load sentinel tiles ---------------------------------
file = open(senTiles_file, "r")
sen_lines = file.readlines()

if dir_name == "Sentinel-12":
    sen_tiles = []
    for line in sen_lines:
        line.strip()
        sen_tiles.append(ast.literal_eval(line))
    sen_tiles = sen_tiles + [sentinel12_pred] 
    bands_dict = {"VH":[], "VV": [], "B11": [], "B12": [], "B2": [], "B3": [], "B4": [], "B5": [], "B6": [], "B7": [], "B8": [], "B8A": []}
    
elif dir_name == "Sentinel-2":
    sen_tiles = sen_lines + [sentinel2_pred]
    sen_tiles = [tile.strip() for tile in sen_tiles]
    bands_dict = {"B11": [], "B12": [], "B2": [], "B3": [], "B4": [], "B5": [], "B6": [], "B7": [], "B8": [], "B8A": []}

else:
    sen_tiles = sen_lines + [sentinel1_pred]
    sen_tiles = [tile.strip() for tile in sen_tiles]
    bands_dict = {"VH":[], "VV": []}

[print(tile) for tile in sen_tiles]

# # ----------------------------- Normalization parameters -------------------------------------------
print("Start with tile calculating normalization parameter for each band")

norm_textfile = os.path.join(output_folder, "normParameter32UPE.txt")

if os.path.exists(norm_textfile):
    bands_scale = json.load(open(norm_textfile))

else:

    for tile_folder in sen_tiles:
        sen_path = getBandPaths(tile_folder, dir_name)

        for band in sen_path:
            band_name = os.path.basename(band).split(".")[0].split("_")[-1]
            bands_dict[band_name].append(load_img_as_array(band))

    bands_scale = {}

    for key, value in bands_dict.items():
        band_flatten = np.concatenate([x.flatten() for x in value]) # one big flattened array of each band 
        c,d = np.nanpercentile(band_flatten, [0.1, 99.9]) # find normalization parameters for each band
        bands_scale[key] = [c,d]

    json.dump(bands_scale, open(norm_textfile, "w"))

print("Found normalization values and start croping, scaling Sentinel tiles")

import pdb 
pdb.set_trace()

# # -------------------- Rasterize PV & Crop Sentinel 2 tiles & Calculate IDX -----------------------
# # Get input data 

# for idx1, tile in enumerate(sen_tiles):

#     tile_name, sen_path, raster_muster = getTileBandsRaster(tile, dir_name)         
#     print("Start with tile: ", tile_name)

#     # Create mask as raster for each sentinel tile
#     mask_path = rasterizeShapefile(raster_muster, solar_path, output_folder, tile_name, col_name="SolarPark")

#     # all paths in one list
#     sen_mask = sen_path + [mask_path] # [B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A (VH VV) MASK]

#     bands_patches = {} # {"B11": [[patch1], [patch2] ..., "B11": [...], ..., "SolarParks": [...]}
    
#     # Patchify all input data -> create smaller patches
#     for idx2, band in enumerate(sen_mask):

#         band_name = os.path.basename(band).split(".")[0].split("_")[-1]
#         print("Start with band: ", band_name)
        
#         raster = rasterio.open(band)
        
#         # resample all bands that do not have a resolution of 10x10m
#         if raster.transform[0] != 10:
#             raster = resampleRaster(band, 10)
#             _,xres,_,_,_,yres = raster.GetGeoTransform()
#             if xres == 10 and abs(yres) == 10:
#                 print("Succesfully resampled band")
#             else:
#                 print("Need to check the resampling again")
#             r_array = raster.ReadAsArray()
#             r_array = np.expand_dims(r_array, axis=0)

#         else:
#             r_array = raster.read()[:,:10980,:10980]

#         r_array = np.moveaxis(r_array, 0, -1)
#         r_array = np.nan_to_num(r_array)
    
#         if idx2 != (len(sen_mask)-1):
                  
#             # 1 and 99 perzentile + [0,1]            
#             a,b = 0,1
#             c,d = bands_scale[band_name]
#             r_array_norm = (b-a)*((r_array-c)/(d-c))+a
#             r_array_norm[r_array_norm > 1] = 1
#             r_array_norm[r_array_norm < 0] = 0
            
#             # if idx1 == 0:

#                 # Min Max Scaling 
#                 # r_array_flat = r_array.flatten()
#                 # r_array_minmax = ((r_array-np.amin(r_array_flat))/(np.amax(r_array_flat)-np.amin(r_array_flat)))

#                 # # 1 and 99 perzentile 
#                 # f,l = np.nanpercentile(r_array, [1,99])
#                 # r_array_1_99 = ((r_array-f)/(l-f))

#                 # # Plot histogram of linear normalization 
#                 # q25, q75 = np.percentile(r_array, [25, 75])
#                 # bin_width = 2 * (q75 - q25) * len(r_array) ** (-1/3)
#                 # bins = round((r_array.max() - r_array.min()) / bin_width)   
#                 # rows, cols = 2, 2
#                 # plt.figure(figsize=(20,20))
#                 # plt.subplot(rows, cols, 1)
#                 # plt.title("Band {} - Original histogram".format(band_name), fontsize = 20)
#                 # plt.hist(r_array.flatten(), bins = bins, color="lightcoral")
#                 # plt.ylabel('Number of pixels', fontsize = 16)
#                 # plt.xlabel('DN', fontsize = 16)
#                 # plt.subplot(rows, cols, 2)
#                 # plt.title("Band {} - MinMax normalization".format(band_name), fontsize = 20)
#                 # plt.hist(r_array_minmax.flatten(), bins = bins, color="lightblue")
#                 # plt.ylabel('Number of pixels', fontsize = 16)
#                 # plt.xlabel('Normalized values', fontsize = 16)
#                 # plt.subplot(rows, cols, 3)
#                 # plt.title("Band {} - Normalization 1st/99th percentile".format(band_name), fontsize = 20)
#                 # plt.hist(r_array_1_99.flatten(), bins = bins, color="lightblue")
#                 # plt.ylabel('Number of pixels', fontsize = 16)
#                 # plt.xlabel('Normalized values', fontsize = 16)
#                 # plt.subplot(rows, cols, 4)
#                 # plt.title("Band {} - Normalization 1st/99th percentile [0,1]".format(band_name), fontsize = 20)
#                 # plt.hist(r_array_norm.flatten(), bins = bins, color="lightblue")
#                 # plt.ylabel('Number of pixels', fontsize = 16)
#                 # plt.xlabel('Normalized values', fontsize = 16)
#                 # plt.savefig(f"{output_folder}/Histo{band_name}_linearNorm.jpg")
                
#         else:
#             r_array_norm = r_array
        
#         bands_patches[band_name] = patchifyRasterAsArray(r_array_norm, patch_size)

#     # Calculate important indizes
#     if indizes:
#         print("Calculating indizes")
#         if dir_name == "Sentinel-12":
#             bands_patches = calculateIndizesSen12(bands_patches)
#         elif dir_name == "Sentinel-2":
#             bands_patches = calculateIndizesSen2(bands_patches)
#         else:
#             bands_patches = calculateIndizesSen1(bands_patches)

#     # Save patches in folder as raster file 
#     # Save trainings data 
#     if idx1 != len(sen_tiles)-1: 
#         savePatchesPV(bands_patches, X_t_out, y_t_out, seed, raster_muster) # save patches for training data 
#     # save prediction data 
#     else:
#         # save patches for prediction - necessary because i want all preprocess work done in one file 
#         savePatchesPV(bands_patches, X_p_out, y_p_out, seed, raster_muster) # save same amount of PV and noPV images
        
#         mask_name = os.path.basename(tile_name).split("_")[1]
#         del bands_patches[mask_name]
#         savePatchesFullImg(bands_patches, p_full_out, tile_name) # save patches of entire sentiel 2 tile for prediction 

#     # Clear memory
#     bands_patches = {}
#     del r_array
#     del raster

## ------------------ Image Augmentation -------------------------------
img_list = glob("{}/*.tif".format("/home/hoehn/data/output/Sentinel-12/crops256/idx/train/img"))
mask_list = glob("{}/*.tif".format("/home/hoehn/data/output/Sentinel-12/crops256/idx/train/mask"))

print(len(img_list))
print(len(mask_list))

img_list.sort()
mask_list.sort()

# X_train, X_test, y_train, y_test = train_test_split(img_list, mask_list, test_size = 0.20, shuffle=True, random_state = seed)

# # Move test data in test directory
# print("Move test data in test directory - make sure that no test data is in the training process")

# for idx in range(len(X_test)):
#     X_test_src = X_test[idx]
#     X_test_dst = "/".join(["test" if i == "train" else i for i in X_test[idx].split(os.sep)]) 
#     y_test_src = y_test[idx]
#     y_test_dst = "/".join(["test" if i == "train" else i for i in y_test[idx].split(os.sep)]) 
#     shutil.move(X_test_src, X_test_dst)
#     shutil.move(y_test_src, y_test_dst)

# print("Files in training/img folder: ", len((os.listdir(os.path.dirname(X_train[0])))))
# print("Files in test/img folder: ", len((os.listdir(os.path.dirname(X_test_dst[0])*len(X_test)))))

##Image Augmentation 
# X_augFolder, y_augFolder = imageAugmentation(X_train, y_train, seed)  
# X_augFolder, y_augFolder = imageAugmentation(img_list, mask_list, seed) 

print("Augmented masks are stored in folder: ", X_augFolder)
print("Augmented images are stored in folder: ", y_augFolder)

    