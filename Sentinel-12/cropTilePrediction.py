import os
import shutil
import yaml
from glob import glob
import rasterio
import numpy as np
from tools_preprocess import resampleRaster, patchifyRasterAsArray, calculateIndizesSen12, savePatchesPredict

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        patch_size = data['model_parameter']['patch_size']
        indizes = data["indizes"]
        sen1_folder = data['prediction']["data"]['Sentinel1']
        sen2_folder = data['prediction']["data"]['Sentinel2']
        sen1_paths = glob("{}/*.tif".format(sen1_folder))
        sen2_paths = glob("{}/*.tif".format(sen2_folder))

tile_name = sen2_folder.split("_")[-1]
output_folder = os.path.join("/home/hoehn/data/prediction", tile_name)

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
    os.mkdir(output_folder)

bands_patches = {}

sentinel_paths = sen1_paths + sen2_paths # VH VV B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A

for idx, band in enumerate(sentinel_paths):

    band_name = os.path.basename(band).split("_")[-1].split(".")[0]
    print("Start patching with band: ", band_name)
    raster = rasterio.open(band)
    
    if raster.transform[0] != 10:  
        raster = resampleRaster(band, 10)
        r_array = raster.ReadAsArray()
        r_array = np.expand_dims(r_array, axis=0)
    else:
        r_array = raster.read()[:,:10980,:10980]
    
    r_array = np.moveaxis(r_array, 0, -1)
    r_array = np.nan_to_num(r_array)
    
    a,b = 0,1
    c,d = np.percentile(r_array, [0.1, 99.9])
    r_array_norm = (b-a)*((r_array-c)/(d-c))+a
    r_array_norm[r_array_norm > 1] = 1
    r_array_norm[r_array_norm < 0] = 0 

    bands_patches[band_name] = patchifyRasterAsArray(r_array_norm, patch_size)

if indizes:
    bands_patches = calculateIndizesSen12(bands_patches)

patches_path = savePatchesPredict(bands_patches, output_folder)
