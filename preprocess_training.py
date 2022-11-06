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
if os.path.exists("config_training.yaml"):
    with open('config_training.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)

        # preprocess for training
        aoi_path = data['shapefiles']['AOI']
        solar_path = data['shapefiles']['Solar']
        sentinel1_name = data['satellite']['Sentinel1']['name']
        sentinel2_name = data['satellite']['Sentinel2']['name']
        sentinel1_pl = data['satellite']['Sentinel1']['processing_level']
        sentinel2_pl = data['satellite']['Sentinel2']['processing_level']
        sentinel1_pt = data['satellite']['Sentinel1']['product_type']
        sentinel2_pt = data['satellite']['Sentinel2']['product_type']
        start_date = data['satellite']['search_criteria']['start_date']
        end_date = data['satellite']['search_criteria']['end_date']
        cloud_cover = data['satellite']['search_criteria']['cloud_cover']
    
        patch_size = data['model']['patch_size']
        output_folder = data["output_folder"]

# get path of sentinel 1 and 2 tiles on codede server
sen2_scenes = ScenceFinderAOI(aoi_path, sentinel2_name, sentinel2_pl, sentinel2_pt, start_date, end_date, cloud_cover)
sen1_scenes = ScenceFinderAOI(aoi_path, sentinel1_name, sentinel1_pl, sentinel1_pt, start_date, end_date, cloud_cover)
sceneList = []
for sen2 in sen2_scenes:
    for sen1 in sen1_scenes:
        sen1_id = str('_'.join(sen1.split("_")[-2:]))
        sen2_id = str('_'.join(sen2.split("_")[-2:]))

        if sen1_id == sen2_id:
            sceneList.append([sen1,sen2])

Sen12_tiles = filterSen12(sceneList, filterDate=False)
Sen12_tiles.sort()

print("Found the following Sentinel 1 and Sentinel 2 scenes")
[print(i) for i in Sen12_tiles]

crop_folder = os.path.join(output_folder, "Sen12_indizes_crops")

# Clean Crops directory in output folder
shutil.rmtree(crop_folder)
os.mkdir(crop_folder)
os.mkdir(os.path.join(crop_folder, "img"))
os.mkdir(os.path.join(crop_folder, "mask"))

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

Sen2_tiles = []

for i in Sen12_tiles:
    Sen2_tiles.append(i[1])

print("Found the following Sentinel 2 scenes")
[print(i) for i in Sen2_tiles]

crop_folder = os.path.join(output_folder, "Sen2_indizes_crops")

# Clean Crops directory in output folder
shutil.rmtree(crop_folder)
os.mkdir(crop_folder)
os.mkdir(os.path.join(crop_folder, "img"))
os.mkdir(os.path.join(crop_folder, "mask"))

# Get input data = Sentinel 2
for tile in Sen2_tiles:
    # get tile name and path for each band
    tile_name = '_'.join(tile.split("_")[-2:])
    sen_path = glob(f"{tile}/*.tif") 
    sen_path.sort() # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A

    print("Start with tile: ", tile_name)

    # Create mask as raster for each sentinel tile
    mask_path = rasterizeShapefile(sen_path[2], solar_path, output_folder, tile_name, col_name="SolarPark")

    # all paths in one list
    sen_mask = sen_path + [mask_path] # [B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A MASK]

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
    bands_patches = calculateIndizesSen2(bands_patches)

    # Save patches in folder as raster file
    images_path, masks_path = savePatchesTrain(bands_patches, crop_folder)

    # Clear memory
    bands_patches = {}
    del r_array
    del raster

# Data augmentation of saved patches
imageAugmentation(images_path, masks_path)
print("---------------------")

