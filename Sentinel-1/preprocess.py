import os
from glob import glob
import yaml
import numpy as np
from tools_preprocess import *
import rasterio
import shutil
import warnings

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
        sentinel1_pred = data['prediction']["data"]['Sentinel1']
        solar_path_pred = data['prediction']["data"]["solar_path"]

        seed = data["seed"]

file = open("/home/hoehn/data/output/Sentinel-1/sentinel1_tiles.txt", "r")

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
    
Sen1_tiles = file.readlines()

Sen1_tiles = Sen1_tiles + [sentinel1_pred] 

[print(tile.strip()) for tile in Sen1_tiles]

# Get input data = Sentinel 2
for idx1,tile in enumerate(Sen1_tiles):

    # remove \n from file path
    tile = tile.strip()
    # get tile name and path for each band
    tile_name = '_'.join(tile.split("_")[-2:])
    sen_path = glob(f"{tile}/*.tif") 
    sen_path.sort() # VH VV 

    print("Start with tile: ", tile_name)

    # Create mask as raster for each sentinel tile
    mask_path = rasterizeShapefile(sen_path[0], solar_path, output_folder, tile_name, col_name="SolarPark")

    # all paths in one list
    sen_mask = sen_path + [mask_path] # [VH VV MASK]

    bands_patches = {} # {"VH": [[patch1], [patch2] ..., "VV": [...], ..., "SolarParks": [...]}
    
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
        #r_array = 10 ** (r_array / 10) # transfer log db values in linear values    
        
        if idx2 != (len(sen_mask)-1):
            
            q25, q75 = np.percentile(r_array, [25, 75])
            bin_width = 2 * (q75 - q25) * len(r_array) ** (-1/3)
            bins = round((r_array.max() - r_array.min()) / bin_width)   
            a,b = 0,1
            c,d = np.percentile(r_array, [0.1, 99.9])
            r_array_norm = (b-a)*((r_array-c)/(d-c))+a
            r_array_norm[r_array_norm > 1] = 1
            r_array_norm[r_array_norm < 0] = 0
            
            rows, cols = 2, 2
            plt.figure(figsize=(18,18))
            plt.subplot(rows, cols, 1)
            plt.imshow(r_array)
            plt.title("{} tile without linear normalization".format(band_name))
            plt.subplot(rows, cols, 2)
            plt.imshow(r_array_norm)
            plt.title("{} tile after linear normalization".format(band_name))
            plt.subplot(rows, cols, 3)
            plt.hist(r_array.flatten(), bins = bins)
            plt.ylabel('Number of values')
            plt.xlabel('DN')
            plt.subplot(rows, cols, 4)
            plt.hist(r_array_norm.flatten(), bins = bins)
            plt.ylabel('Number of values')
            plt.xlabel('DN')
            plt.savefig('hist1.png')
            
        else:
            r_array_norm = r_array
        
        bands_patches[band_name] = patchifyRasterAsArray(r_array_norm, patch_size)
    
    # Calculate important indizes
    if indizes:
        bands_patches = calculateIndizesSen1(bands_patches)

    # Save patches in folder as raster file
    if idx1 != len(Sen1_tiles)-1: 
        images_path, masks_path = savePatchesTrain(bands_patches, crop_folder, seed)
    else:
        bands_patches = calculateIndizesSen1(bands_patches)
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

# # Data augmentation of saved patches
imageAugmentation(images_path, masks_path, seed)
print("---------------------")

# check equal size of mask and img dir
images_path = "/home/hoehn/data/output/Sentinel-1/crops/idx/img"
mask_path = "/home/hoehn/data/output/Sentinel-1/crops/idx/mask"

print("Equal size of directories for img/mask crops:", len(os.listdir(images_path)) == len(os.listdir(mask_path)))

# Check that all values are between 0 and 1 
# Check equal shape for all images
masks = os.listdir(mask_path)
countNr, countSh, countVal= 0,0,0
maskNR = []
imgNR = []
[maskNR.append(i.split("_")[2]) for i in os.listdir(mask_path)]
[imgNR.append(i.split("_")[2]) for i in os.listdir(images_path)]
maskNR.sort()
imgNR.sort()

for path in os.listdir(images_path):     
    r = load_img_as_array(os.path.join(images_path, path))
    if r.shape == (128,128,3):
        countSh += 1
    r_flat = np.concatenate(r).flatten()
    result = np.all(r_flat <= 1)
    result2 = np.all(r_flat >= 0)
    if result and result2:
        countVal += 1

print("test 1 {}".format(maskNR == imgNR))
print("test 2 {}".format(countSh == countVal))


