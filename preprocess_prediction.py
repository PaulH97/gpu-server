import os
import numpy as np
import rasterio
import yaml
from tools_preprocess import resampleRaster,patchifyRasterAsArray,savePatchesPredict
from glob import glob

# Read data from config file
if os.path.exists("config_prediction.yaml"):
    with open('config_prediction.yaml') as f:
        
        data = yaml.load(f, Loader=yaml.FullLoader)

        if data['model']['Unet_Sen1_Sen2']:

            sentinel1_folder = data['data_source']['Sentinel1']
            sentinel2_folder = data['data_source']['Sentinel2']
            sentinel_paths = glob("{}/*.tif".format(sentinel2_folder)) + glob("{}/*.tif".format(sentinel1_folder))
            sentinel_paths.sort() 

        elif data['model']['Unet_Sen2']:

            sentinel2_folder = data['data_source']['Sentinel2']
            sentinel_paths = glob("{}/*.tif".format(sentinel2_folder))
            sentinel_paths.sort() 

        patch_size = data['model']['patch_size']
        output_folder = data["output_folder"]
        
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

patches_path = savePatchesPredict(bands_patches, output_folder)
