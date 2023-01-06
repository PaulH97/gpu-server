import os
import yaml
from glob import glob
import rasterio
import numpy as np
from tools_preprocess import resampleRaster, patchifyRasterAsArray, calculateIndizesSen2, savePatchesPredict

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        patch_size = data['model_parameter']['patch_size']
        indizes = data["indizes"]
        tile_folder = data['prediction']["data"]['Sentinel2']
        sentinel_paths = glob("{}/*.tif".format(tile_folder))

tile_name = tile_folder.split("_")[-1]
output_folder = os.path.join("/home/hoehn/data/prediction", tile_name)
os.mkdir(output_folder)

bands_patches = {}

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
    bands_patches = calculateIndizesSen2(bands_patches)

patches_path = savePatchesPredict(bands_patches, output_folder)
