import os
from glob import glob
import yaml
import numpy as np
from tools_preprocess import *
import rasterio
from matplotlib import pyplot as plt
import shutil
import warnings
import ast

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        # get data for preprocessing
        solar_path = data['shapefiles']['Solar']    
        patch_size = data['model_parameter']['patch_size']
        output_folder = data["output_folder"]
        indizes = data["indizes"]

        # data for prediction
        sentinel12_pred = data['prediction']["data"]['Sentinel2']
        solar_path_pred = data['prediction']["data"]["solar_path"]

        seed = data["seed"]

crop_folder = os.path.join(output_folder, "crops")

if indizes:
    crop_folder = os.path.join(crop_folder, "idx")
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
    crop_folder = os.path.join(crop_folder, "no_idx")
    if os.path.exists(crop_folder):
        shutil.rmtree(crop_folder)
        os.mkdir(crop_folder)
        os.mkdir(os.path.join(crop_folder, "img"))
        os.mkdir(os.path.join(crop_folder, "mask"))
    else:
        os.mkdir(crop_folder)
        os.mkdir(os.path.join(crop_folder, "img"))
        os.mkdir(os.path.join(crop_folder, "mask"))

pred_cfolder = os.path.join(output_folder, "prediction", "crops")
if os.path.exists(pred_cfolder):
    shutil.rmtree(pred_cfolder)
    os.mkdir(pred_cfolder)
    os.mkdir(os.path.join(pred_cfolder, "full_img"))
    os.mkdir(os.path.join(pred_cfolder, "img"))
    os.mkdir(os.path.join(pred_cfolder, "mask"))
else:
    os.mkdir(pred_cfolder)
    os.mkdir(os.path.join(pred_cfolder, "full_img"))
    os.mkdir(os.path.join(pred_cfolder, "img"))
    os.mkdir(os.path.join(pred_cfolder, "mask"))
    
file = open("/home/hoehn/data/output/Sentinel-12/sentinel12_tiles.txt", "r")
lines = file.readlines()

Sen12_tiles = []

for line in lines:
    line.strip()
    Sen12_tiles.append(ast.literal_eval(line))

Sen12_tiles = Sen12_tiles + [sentinel12_pred]

[print(tile) for tile in Sen12_tiles]

# Get input data = Sentinel 1 + Sentinel 2 
for idx1, tile in enumerate(Sen12_tiles):

    if idx1 == len(Sen12_tiles)-1:
        # get tile name and path for each band
        tile_name = '_'.join(tile.split("_")[-2:])
        sen_path = glob(f"{tile}/*.tif") 
        sen_path.sort() # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A
    else:
        # get tile name and path for each band
        tile_name = '_'.join(tile[0].split("_")[-2:])
        sen_path =  glob(f"{tile[0]}/*.tif") + glob(f"{tile[1]}/*.tif")
        sen_path.sort() # VH VV B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A 

    print("Start with tile: ", tile_name)

    # Create mask as raster for each sentinel tile
    mask_path = rasterizeShapefile(sen_path[4], solar_path, output_folder, tile_name, col_name="SolarPark")

    # all paths in one list
    sen_mask = sen_path + [mask_path] # [VH VV B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A MASK]

    bands_patches = {} # {"B11": [[patch1], [patch2] ..., "B11": [...], ..., "SolarParks": [...]}
    
    # Patchify all input data -> create smaller patches
    for idx2, band in enumerate(sen_mask):

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
    
        if idx2 != (len(sen_mask)-1):
            
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
    
    # Save patches in folder as raster file
    if idx1 != len(Sen12_tiles)-1:     
        # Calculate important indizes
        if indizes:
            bands_patches = calculateIndizesSen12(bands_patches)
        images_path, masks_path = savePatchesTrain(bands_patches, crop_folder, seed)
    else:
        bands_patches = calculateIndizesSen2(bands_patches)
        images_path_pd, masks_path_pd = savePatchesTrain(bands_patches, pred_cfolder, seed)
        print("Saved crops for prediciton in:{}".format(images_path_pd))
        print("Saved crops for prediciton in:{}".format(masks_path_pd))
        mask_name = os.path.basename(tile_name).split("_")[1]
        del bands_patches[mask_name]
        savePatchesPredict(bands_patches, pred_cfolder)

    # Clear memory
    bands_patches = {}
    del r_array
    del raster

# Data augmentation of saved patches
print(images_path)
print(masks_path)
imageAugmentation(images_path, masks_path, seed)
print("---------------------")


