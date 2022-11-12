import os
from glob import glob
import yaml
import numpy as np
from tools_preprocess import *
import rasterio
from matplotlib import pyplot as plt
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        # get data for preprocessing
        solar_path = data['shapefiles']['Solar']    
        patch_size = data['model_parameter']['patch_size']
        output_folder = data["output_folder"]
        indizes = data["indizes"]

crop_folder = os.path.join(output_folder, "crops")

if indizes:
    crop_folder = os.path.join(crop_folder, "indizes")
    if os.path.exists(crop_folder):
        shutil.rmtree(crop_folder)
        os.mkdir(crop_folder)
        os.mkdir(os.path.join(crop_folder, "img"))
        os.mkdir(os.path.join(crop_folder, "mask"))
    else:
        os.mkdir(crop_folder)
        os.mkdir(os.path.join(crop_folder, "img"))
        os.mkdir(os.path.join(crop_folder, "mask"))
else:
    crop_folder = os.path.join(crop_folder, "no_indizes")
    if os.path.exists(crop_folder):
        shutil.rmtree(crop_folder)
        os.mkdir(crop_folder)
        os.mkdir(os.path.join(crop_folder, "img"))
        os.mkdir(os.path.join(crop_folder, "mask"))
    else:
        os.mkdir(crop_folder)
        os.mkdir(os.path.join(crop_folder, "img"))
        os.mkdir(os.path.join(crop_folder, "mask"))
    
file = open("/home/hoehn/data/output/Sentinel-2/sentinel2_tiles.txt", "r")
Sen12_tiles = file.readlines()

# Get input data = Sentinel 1 + Sentinel 2 
for tile in Sen12_tiles:
    # get tile name and path for each band
    tile_name = '_'.join(tile[0].split("_")[-2:])
    sen_path = glob(f"{tile[1]}/*.tif") + glob(f"{tile[0]}/*.tif") 
    sen_path.sort() # VH VV B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A 

    print("Start with tile: ", tile_name)

    # Create mask as raster for each sentinel tile
    mask_path = rasterizeShapefile(sen_path[2], solar_path, output_folder, tile_name, col_name="SolarPark")

    # all paths in one list
    sen_mask = sen_path + [mask_path] # [VH VV B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A MASK]

    bands_patches = {} # {"B11": [[patch1], [patch2] ..., "B11": [...], ..., "SolarParks": [...]}
    
    # Patchify all input data -> create smaller patches
    for idx, band in enumerate(sen_mask):

        band_name = os.path.basename(band).split(".")[0].split("_")[-1]
        print("Start with band: ", band_name)
        
        raster = rasterio.open(band)

        if raster.transform[0] != 10:
            raster = resampleRaster(band, 10)
            r_array = raster.ReadAsArray()
            r_array = np.expand_dims(r_array, axis=0)
        else:
            r_array = raster.read()[:,:10980,:10980]

        r_array = np.moveaxis(r_array, 0, -1)
        r_array = np.nan_to_num(r_array)
    
        if idx != (len(sen_mask)-1):
            
            q25, q75 = np.percentile(r_array, [25, 75])
            bin_width = 2 * (q75 - q25) * len(r_array) ** (-1/3)
            bins = round((r_array.max() - r_array.min()) / bin_width)   
        
            a,b = 0,1
            c,d = np.percentile(r_array, [0.1, 99.9])
            r_array_norm = (b-a)*((r_array-c)/(d-c))+a
            r_array_norm[r_array_norm > 1] = 1
            r_array_norm[r_array_norm < 0] = 0
            
            # rows, cols = 2, 2
            # plt.figure(figsize=(18,18))
            # plt.subplot(rows, cols, 1)
            # plt.imshow(r_array)
            # plt.title("{} tile without linear normalization".format(band_name))
            # plt.subplot(rows, cols, 2)
            # plt.imshow(r_array_norm)
            # plt.title("{} tile after linear normalization".format(band_name))
            # plt.subplot(rows, cols, 3)
            # plt.hist(r_array.flatten(), bins = bins)
            # plt.ylabel('Number of values')
            # plt.xlabel('DN')
            # plt.subplot(rows, cols, 4)
            # plt.hist(r_array_norm.flatten(), bins = bins)
            # plt.ylabel('Number of values')
            # plt.xlabel('DN')
            # plt.show() 

        else:
            r_array_norm = r_array
        
        bands_patches[band_name] = patchifyRasterAsArray(r_array_norm, patch_size)
    
    # Calculate important indizes
    if indizes:
        bands_patches = calculateIndizesSen12(bands_patches)

    # Save patches in folder as raster file
    images_path, masks_path = savePatchesTrain(bands_patches, crop_folder)

    # Clear memory
    bands_patches = {}
    del r_array
    del raster

# Data augmentation of saved patches
imageAugmentation(images_path, masks_path)
print("---------------------")
