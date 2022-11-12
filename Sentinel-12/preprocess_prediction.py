import os
import numpy as np
import rasterio
import yaml
from tools_preprocess import resampleRaster,patchifyRasterAsArray,savePatchesPredict, calculateIndizesSen12, calculateIndizesSen2, rasterizeShapefile
from glob import glob

# Read data from config file
if os.path.exists("config_prediction.yaml"):
    with open('config_prediction.yaml') as f:
        
        data = yaml.load(f, Loader=yaml.FullLoader)

        sentinel1_folder = data['data_source']['Sentinel1']
        sentinel2_folder = data['data_source']['Sentinel2']
        solar_path = data["solar"]["path"]

        model_name = data['model']['path'].split("/")[-1].split("_")[0]
        indizesSen12 = False
        indizesSen2 = False

        if "Sen12" in model_name:
            sentinel_paths = glob("{}/*.tif".format(sentinel2_folder)) + glob("{}/*.tif".format(sentinel1_folder))
            sentinel_paths.sort()
            if "indizes" in model_name:
                indizesSen12 = True       
        else:
            sentinel_paths = glob("{}/*.tif".format(sentinel2_folder))
            sentinel_paths.sort()
            if "indizes" in model_name:
                indizesSen2 = True 
        
        patch_size = data['model']['patch_size']
        output_folder = data["output_folder"]
        
bands_patches = {}

# print(sentinel_paths)

# for idx, band in enumerate(sentinel_paths):

#     band_name = os.path.basename(band).split("_")[-1].split(".")[0]
#     print("Start patching with band: ", band_name)
#     raster = rasterio.open(band)

#     if raster.transform[0] != 10:  
#         raster = resampleRaster(band, 10)
#         r_array = raster.ReadAsArray()
#         r_array = np.expand_dims(r_array, axis=0)
#     else:
#         r_array = raster.read()[:,:10980,:10980]
    
#     r_array = np.moveaxis(r_array, 0, -1)
#     r_array = np.nan_to_num(r_array)
    
#     a,b = 0,1
#     c,d = np.percentile(r_array, [0.1, 99.9])
#     r_array_norm = (b-a)*((r_array-c)/(d-c))+a
#     r_array_norm[r_array_norm > 1] = 1
#     r_array_norm[r_array_norm < 0] = 0 

#     bands_patches[band_name] = patchifyRasterAsArray(r_array_norm, patch_size)

# if indizesSen12:
#     bands_patches = calculateIndizesSen12(bands_patches)
# else: 
#     bands_patches = calculateIndizesSen2(bands_patches)

# patches_path = savePatchesPredict(bands_patches, output_folder)

# Create mask as raster for each sentinel tile
band_example = glob("{}/*B2.tif".format(sentinel2_folder))[0]
tile_name = sentinel2_folder.split("/")[-2]
mask_path = rasterizeShapefile(band_example, solar_path, output_folder, tile_name, col_name="SolarPark")
print("Created mask file in: ", mask_path)
