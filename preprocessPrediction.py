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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
GDAL_DISABLE_READDIR_ON_OPEN=True
# -------------------- Load data -------------------------------
# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        # get data for preprocessing
        solar_path = data['shapefiles']['Solar_predict']    
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
        
        sen_tiles = [sentinel12_pred] # CHANGE HERE

# # -------------------- Build folder structure ------------------------
crop_name = "prediction/32UPE/" + str(patch_size)
crop_folder = os.path.join(output_folder, crop_name)

if indizes:
    crop_folder = os.path.join(crop_folder, "idx")
    X_t_out, y_t_out, X_p_out, y_p_out, p_full_out = rebuildCropFolder(crop_folder)
else:
    crop_folder = os.path.join(crop_folder, "no_idx")
    X_t_out, y_t_out, X_p_out, y_p_out, p_full_out = rebuildCropFolder(crop_folder)

# # ----------------------------- Normalization parameters -------------------------------------------
print("Start with tile calculating normalization parameter for each band")
norm_textfile = os.path.join(output_folder, "normParameter.txt")

if os.path.exists(norm_textfile):
    bands_scale = json.load(open(norm_textfile))

# -------------------- Rasterize PV & Crop Sentinel 2 tiles & Calculate IDX -----------------------
# Get input data 

for idx1, tile in enumerate(sen_tiles):

    tile_name, sen_path, raster_muster = getTileBandsRaster(tile, dir_name)         
    print("Start with tile: ", tile_name)
    # Create mask as raster for each sentinel tile
    mask_path = rasterizeShapefile(raster_muster, solar_path, output_folder, tile_name, col_name="SolarPark")

    # all paths in one list
    sen_mask = sen_path + [mask_path] # [B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A (VH VV) MASK]
    bands_patches = {} # {"B11": [[patch1], [patch2] ..., "B11": [...], ..., "SolarParks": [...]}
    
    # Patchify all input data -> create smaller patches
    for idx2, band in enumerate(sen_mask):

        band_name = os.path.basename(band).split(".")[0].split("_")[-1]
        print("Start with band: ", band_name)
        
        raster = rasterio.open(band)
        
        # resample all bands that do not have a resolution of 10x10m
        if raster.transform[0] != 10:
            raster = resampleRaster(band, 10)
            _,xres,_,_,_,yres = raster.GetGeoTransform()
            if xres == 10 and abs(yres) == 10:
                print("Succesfully resampled band")
            else:
                print("Need to check the resampling again")
            r_array = raster.ReadAsArray()
            r_array = np.expand_dims(r_array, axis=0)

        else:
            r_array = raster.read()[:,:10980,:10980]

        r_array = np.moveaxis(r_array, 0, -1)
        r_array = np.nan_to_num(r_array)
    
        if idx2 != (len(sen_mask)-1):
                  
            # 1 and 99 perzentile + [0,1]            
            a,b = 0,1
            c,d = bands_scale[band_name]
            r_array_norm = (b-a)*((r_array-c)/(d-c))+a
            r_array_norm[r_array_norm > 1] = 1
            r_array_norm[r_array_norm < 0] = 0
                            
        else:
            r_array_norm = r_array
        
        bands_patches[band_name] = patchifyRasterAsArray(r_array_norm, patch_size)

    # Calculate important indizes
    if indizes:
        print("Calculating indizes")
        if dir_name == "Sentinel-12":
            bands_patches = calculateIndizesSen12(bands_patches)
        elif dir_name == "Sentinel-2":
            bands_patches = calculateIndizesSen2(bands_patches)
        else:
            bands_patches = calculateIndizesSen1(bands_patches)

    # save patches for prediction - necessary because i want all preprocess work done in one file 
    savePatchesPV(bands_patches, X_p_out, y_p_out, seed, raster_muster) # save same amount of PV and noPV images
    
    mask_name = os.path.basename(tile_name).split("_")[1]
    del bands_patches[mask_name]
    savePatchesFullImg(bands_patches, p_full_out, tile_name) # save patches of entire sentiel 2 tile for prediction 

    # Clear memory
    bands_patches = {}
    del r_array
    del raster