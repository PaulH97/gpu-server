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
        sentinel2_pred = data['prediction']["data"]['Sentinel2']
        solar_path_pred = data['prediction']["data"]["solar_path"]

        seed = data["seed"]

crop_folder = os.path.join(output_folder, "crops")
if indizes:
    crop_folder = os.path.join(crop_folder, "idx")
    rebuildCropFolder(crop_folder)
else:
    crop_folder = os.path.join(crop_folder, "no_idx")
    rebuildCropFolder(crop_folder)

pred_cfolder = os.path.join(output_folder, "prediction", "crops")
rebuildPredFolder(pred_cfolder)
    
file = open("/home/hoehn/data/output/Sentinel-2/sentinel2_tiles.txt", "r")
Sen2_tiles = file.readlines()

Sen2_tiles = Sen2_tiles + [sentinel2_pred] # Sentinel-2 training tiles + Sentinel-2 prediction tile

[print(tile.strip()) for tile in Sen2_tiles]

# Get input data = Sentinel 2
for idx1,tile in enumerate(Sen2_tiles):

    # remove \n from file path
    tile = tile.strip()
    # get tile name and path for each band
    tile_name = '_'.join(tile.split("_")[-2:])
    sen_path = glob(f"{tile}/*.tif") 
    sen_path.sort() # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A

    # raster muster for rasterize shps + georeferenzing patches
    raster_muster = sen_path[2]
    print("Start with tile: ", tile_name)

    # Create mask as raster for each sentinel tile
    mask_path = rasterizeShapefile(raster_muster, solar_path, output_folder, tile_name, col_name="SolarPark")

    # all paths in one list
    sen_mask = sen_path + [mask_path] # [B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A MASK]

    bands_patches = {} # {"B11": [[patch1], [patch2] ..., "B11": [...], ..., "SolarParks": [...]}
    
    # Patchify all input data -> create smaller patches
    for idx2, band in enumerate(sen_mask):

        band_name = os.path.basename(band).split(".")[0].split("_")[-1]
        print("Start with band: ", band_name)
        
        raster = rasterio.open(band)
        
        # resample all bands that do not have a resolution of 10x10m
        if raster.transform[0] != 10:
            raster = resampleRaster(band, 10)
            r_array = raster.ReadAsArray()
            r_array = np.expand_dims(r_array, axis=0) # (10980,10980,1)
        else:
            r_array = raster.read()[:,:10980,:10980]

        r_array = np.moveaxis(r_array, 0, -1)
        r_array = np.nan_to_num(r_array)
    
        if idx2 != (len(sen_mask)-1):
                  
            a,b = 0,1
            c,d = np.percentile(r_array, [0.01, 99.9])
            r_array_norm = (b-a)*((r_array-c)/(d-c))+a
            r_array_norm[r_array_norm > 1] = 1
            r_array_norm[r_array_norm < 0] = 0

            # q25, q75 = np.percentile(r_array, [25, 75])
            # bin_width = 2 * (q75 - q25) * len(r_array) ** (-1/3)
            # bins = round((r_array.max() - r_array.min()) / bin_width)   
            
            # rows, cols = 2, 2
            # plt.figure(figsize=(18,18))
            # plt.subplot(rows, cols, 1)
            # plt.imshow(r_array)
            # plt.title("{} tile without normalization".format(band_name))
            # plt.subplot(rows, cols, 2)
            # plt.imshow(r_array_norm)
            # plt.title("{} tile after normalization".format(band_name))
            # plt.subplot(rows, cols, 3)
            # plt.hist(r_array.flatten(), bins = bins)
            # plt.ylabel('Number of values')
            # plt.xlabel('DN')
            # plt.subplot(rows, cols, 4)
            # plt.hist(r_array_norm.flatten(), bins = bins)
            # plt.ylabel('Number of values')
            # plt.xlabel('DN')
            # plt.savefig("Histo.jpg")

        else:
            r_array_norm = r_array
        
        bands_patches[band_name] = patchifyRasterAsArray(r_array_norm, patch_size)
    
    # Calculate important indizes
    if indizes:
        bands_patches = calculateIndizesSen2(bands_patches)

    # Save patches in folder as raster file
    if idx1 != len(Sen2_tiles)-1: 
        images_path, masks_path = savePatchesTrain(bands_patches, crop_folder, seed, raster_muster) # save patches for training data 
    else:
        # save patches for prediction - necessary because i want all preprocess work done in one file 
        mask_name = os.path.basename(tile_name).split("_")[1]
        del bands_patches[mask_name]
        savePatchesPredict(bands_patches, pred_cfolder) # save patches of entire sentiel 2 tile for prediction 

    # Clear memory
    bands_patches = {}
    del r_array
    del raster

img_list = glob("{}/*.tif".format(images_path))
mask_list = glob("{}/*.tif".format(masks_path))

img_list.sort()
mask_list.sort()

X_train, X_test, y_train, y_test = train_test_split(img_list, mask_list, test_size = 0.10, shuffle=True, random_state = seed)

# Move test data in test directory
for idx in range(len(X_test)):
    X_test_src = X_test[idx]
    X_test_dst = "/".join(["test" if i == "train" else i for i in X_test[idx].split(os.sep)]) 
    y_test_src = y_test[idx]
    y_test_dst = "/".join(["test" if i == "train" else i for i in y_test[idx].split(os.sep)]) 
    shutil.move(X_test_src, X_test_dst)
    shutil.move(y_test_src, y_test_dst)

imageAugmentation(X_train, y_train, seed)