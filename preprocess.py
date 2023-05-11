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
        seed = data["seed"]
        # name of output directory -> see if Sentinel-1, Sentinel-2 or Sentinel-12
        dir_name = output_folder.split("/")[-1]
        tiles_file = data["satellite"]["tiles_path"]
        idx_dir = os.path.join(output_folder, "indizes")

# # # -------------------- Build folder structure ------------------------
# crop_name = "crops" + str(patch_size)
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

# # ##------------------- Load sentinel tiles ---------------------------------
# file = open(senTiles_file, "r")
# sen_lines = file.readlines()

# if dir_name == "Sentinel-12":
#     sen_tiles = []
#     for line in sen_lines:
#         line.strip()
#         sen_tiles.append(ast.literal_eval(line))
#     sen_tiles = sen_tiles 
#     bands_dict = {"VH":[], "VV": [], "B11": [], "B12": [], "B2": [], "B3": [], "B4": [], "B5": [], "B6": [], "B7": [], "B8": [], "B8A": [], "NDVI": [], "NDWI": [], "CR": []}
    
# elif dir_name == "Sentinel-2":
#     sen_tiles = sen_lines 
#     sen_tiles = [tile.strip() for tile in sen_tiles]
#     bands_dict = {"B11": [], "B12": [], "B2": [], "B3": [], "B4": [], "B5": [], "B6": [], "B7": [], "B8": [], "B8A": [], "NDVI": [], "NDWI": []}

# else:
#     sen_tiles = sen_lines
#     sen_tiles = [tile.strip() for tile in sen_tiles]
#     bands_dict = {"VH":[], "VV": [], "CR":[]}

# [print(tile) for tile in sen_tiles]

# # ----------------------------------------------------------------------------------------------------
# # Calculate important indizes
# # if indizes:   
# #     for tile_folder in sen_tiles:
# #         tile_id = tile_folder.split("_")[-1]
# #         print("Calculating indizes for tile: ", tile_id )
# #         sen_paths = getBandPaths(tile_folder, dir_name)
# #         out_dir = os.path.join(output_folder, "indizes")
        
# #         if dir_name == "Sentinel-12":
# #             indizes_dir = calculateIndizesSen12(sen_paths, tile_id, bands_dict, out_dir)
# #         elif dir_name == "Sentinel-2":
# #             indizes_dir = calculateIndizesSen2(sen_paths, tile_id, bands_dict, out_dir)
# #         else:
# #             indizes_dir = calculateIndizesSen1(sen_paths, tile_id, bands_dict, out_dir)

# # # ----------------------------- Normalization parameters -------------------------------------------
# print("Start with tile calculating normalization parameter for each band")

# norm_textfile = os.path.join(output_folder, "normParameter.txt")

# if os.path.exists(norm_textfile):
#     bands_scale = json.load(open(norm_textfile))

# else:

#     for tile_folder in sen_tiles:
#         sen_paths = getBandPaths(tile_folder, dir_name)

#         for band in sen_paths:
#             band_name = os.path.basename(band).split(".")[0].split("_")[-1]
#             bands_dict[band_name].append(load_img_as_array(band))

#     bands_scale = {}

#     for key, value in bands_dict.items():
#         band_flatten = np.concatenate([x.flatten() for x in value]) # one big flattened array of each band 
#         c,d = np.nanpercentile(band_flatten, [0.1, 99.9]) # find normalization parameters for each band
#         bands_scale[key] = [c,d]

#     json.dump(bands_scale, open(norm_textfile, "w"))

# print("Found normalization values and start croping, scaling Sentinel tiles")

# # # -------------------- Rasterize PV & Crop Sentinel 2 tiles & Calculate IDX -----------------------
# # # Get input data 
# for idx1, tile in enumerate(sen_tiles):

#     tile_name, sen_path, raster_muster = getTileBandsRaster(tile, dir_name, idx_dir)         
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
#         if raster.transform[0] != 10 and band_name not in ["ndvi", "ndwi", "cr"]:
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
#             c,d = bands_scale[band_name.upper()]
#             r_array_norm = (b-a)*((r_array-c)/(d-c))+a
#             r_array_norm[r_array_norm > 1] = 1
#             r_array_norm[r_array_norm < 0] = 0
            
#             print("Successfully normalized band")
              
#         else:
#             r_array_norm = r_array
        
#         bands_patches[band_name] = patchifyRasterAsArray(r_array_norm, patch_size)
            
#     # Save patches in folder as raster file 
#     # Save trainings data 
#     savePatchesPV(bands_patches, X_t_out, y_t_out, seed, raster_muster) # save patches for training data 

#     # Clear memory
#     bands_patches = {}
#     del r_array
#     del raster

# ## ------------------ Image Augmentation -------------------------------
# img_list = glob("{}/*.tif".format(X_t_out))
# mask_list = glob("{}/*.tif".format(y_t_out))

# print(len(img_list))
# print(len(mask_list))

# img_list.sort()
# mask_list.sort()

# import pdb 
# pdb.set_trace()

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

#Image Augmentation 
# X_augFolder, y_augFolder = imageAugmentation(X_train, y_train, seed)  
img_list = glob("{}/*.tif".format("/home/hoehn/data/output/Sentinel-1/crops128/idx/train/img"))
mask_list = glob("{}/*.tif".format("/home/hoehn/data/output/Sentinel-1/crops128/idx/train/mask"))
img_list.sort()
mask_list.sort()

X_augFolder, y_augFolder = imageAugmentation(img_list, mask_list, seed) 

print("Augmented masks are stored in folder: ", X_augFolder)
print("Augmented images are stored in folder: ", y_augFolder)

    