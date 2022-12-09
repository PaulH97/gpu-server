import rasterio
from patchify import patchify,unpatchify
from rasterio import features
import numpy as np
import geopandas as gdp
from osgeo import gdal
import os
import tifffile as tiff
import random
from matplotlib import pyplot as plt
import requests
import geopandas as gpd

np.seterr(divide='ignore', invalid='ignore')

def load_img_as_array(path):
    # read img as array 
    img_array = rasterio.open(path).read()
    img_array = np.moveaxis(img_array, 0, -1)
    #img_array = np.nan_to_num(img_array)
    return img_array

def resampleRaster(raster_path, resolution):
    
    raster = gdal.Open(raster_path)
    #ds = gdal.Warp(output_file, raster, xRes=resolution, yRes=resolution, resampleAlg="bilinear", format="GTiff")
    ds = gdal.Warp('', raster, xRes=resolution, yRes=resolution, resampleAlg="bilinear", format="VRT")

    return ds

def rasterizeShapefile(raster_path, vector_path, output_folder, tile_name, col_name):
    
    raster = rasterio.open(raster_path)
    vector = gdp.read_file(vector_path)
    vector = vector.to_crs(raster.crs)
    
    geom_value = ((geom,value) for geom, value in zip(vector.geometry, vector[col_name]))
    crs    = raster.crs
    transform = raster.transform

    #tile_name = '_'.join(tile_name.split("_")[-2:])
    output_folder = os.path.join(output_folder, "masks")
    r_out = os.path.join(output_folder, os.path.basename(vector_path).split(".")[0] + "_" + tile_name +".tif")

    with rasterio.open(r_out, 'w+', driver='GTiff',
            height = raster.height, width = raster.width,
            count = 1, dtype="int16", crs = crs, transform=transform) as rst:
        out_arr = rst.read(1)
        rasterized = features.rasterize(shapes=geom_value, fill=0, out=out_arr, transform = rst.transform)
        rst.write_band(1, rasterized)
    rst.close()
    return r_out

def patchifyRasterAsArray(array, patch_size):

    patches = patchify(array, (patch_size, patch_size, 1), step=patch_size)    
    patchX = patches.shape[0]
    patchY = patches.shape[1]
    result = []
    #scaler = MinMaxScaler()

    for i in range(patchX):
        for j in range(patchY):
            single_patch_img = patches[i,j,:,:]
            #single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
            result.append(single_patch_img)
    
    return result

def savePatchesTrain(patches, output_folder, seed, raster_muster):

    mask_out = os.path.join(output_folder, "mask") 
    img_out = os.path.join(output_folder, "img") 
    
    mask_dict = {k: v for k, v in patches.items() if k.startswith("3")}
    for k in mask_dict.keys():
        mask_name = k
    band_names_dict = {k: v for k, v in patches.items() if not k.startswith("3")}
    band_names = []
    for k in band_names_dict.keys():
        band_names.append(k)

    band_names.sort()
    idx_noPV = []
    countPV = 0

    r_muster = rasterio.open(raster_muster)
    r_transform = r_muster.transform
    r_crs = r_muster.crs
    
    row = 0
    col = 0
    col_reset = True

    for idx, mask in enumerate(patches[mask_name]):
        
        idx_t = idx + 1
        if col_reset:
            col = 0
            col_reset = False
        else:
            col += 128
   	
        xs, ys = rasterio.transform.xy(r_transform, row, col, offset='ul')
        new_transform = rasterio.transform.from_origin(xs, ys, r_transform[0], -(r_transform[4]))

        if idx_t % 85 == 0:
            row = int(idx_t / 85) * 128
            col_reset = True   

        mask_flat = np.concatenate(mask).flatten()

        if 1 in mask_flat:
                
            tiff.imwrite(os.path.join(mask_out, f'{mask_name}_mask_{idx}_pv.tif'), mask)

            final = rasterio.open(os.path.join(img_out, f'{mask_name}_img_{idx}_pv.tif'),'w', driver='Gtiff',
                            width=patches[band_names[0]][0].shape[0], height=patches[band_names[0]][0].shape[1],
                            count=len(band_names),
                            crs=r_crs,
                            transform=new_transform,
                            dtype=rasterio.float64)

            for band_nr, band_name in enumerate(band_names):
                final.write(patches[band_name][idx][:,:,0],band_nr+1)
            final.close()
            countPV += 1
        else:
            idx_noPV.append(idx)
    
    random.seed(seed)
    random_idx = random.sample(idx_noPV, countPV)
    
    row = 0
    col = 0
    col_reset = True

    for idx, mask in enumerate(patches[mask_name]):

        idx_t = idx + 1
        if col_reset:
            col = 0
            col_reset = False
        else:
            col += 128
    	
        xs, ys = rasterio.transform.xy(r_transform, row, col, offset='ll')
        new_transform = rasterio.transform.from_origin(xs, ys, r_transform[0], -(r_transform[4]))

        if idx_t % 85 == 0:
            row = int(idx_t / 85) * 128
            col_reset = True     

        if idx in random_idx:

            tiff.imwrite(os.path.join(mask_out, f'{mask_name}_mask_{idx}_nopv.tif'), mask)

            final = rasterio.open(os.path.join(img_out, f'{mask_name}_img_{idx}_nopv.tif'),'w', driver='Gtiff',
                            width=patches[band_names[0]][0].shape[0], height=patches[band_names[0]][0].shape[1],
                            count=len(band_names),
                            crs=r_crs,
                            transform=r_transform,
                            dtype=rasterio.float64)

            for band_nr, band_name in enumerate(band_names):
                final.write(patches[band_name][idx][:,:,0],band_nr+1)
            final.close()
        
    return img_out, mask_out

def savePatchesPredict(patches, output_folder):

    img_out = os.path.join(output_folder, "full_img") 
    
    if not os.path.exists(img_out):
        os.makedirs(img_out)
    else:
        for f in os.listdir(img_out):
            os.remove(os.path.join(img_out, f))
    
    band_names = list(patches.keys())
    band_names.sort()

    for idx in range(len(patches[band_names[0]])):

            final = rasterio.open(os.path.join(img_out, f'img_{idx}.tif'),'w', driver='Gtiff',
                            width=patches[band_names[0]][0].shape[0], height=patches[band_names[0]][0].shape[1],
                            count=len(band_names),
                            dtype=rasterio.float64)

            for band_nr, band_name in enumerate(band_names):
                final.write(patches[band_name][idx][:,:,0],band_nr+1)
            final.close()

    return 

def calculateIndizesSen12(bands_patches):

    cr_list, ndvi_list, ndwi_list = [], [], []
    cr_list_norm, ndvi_list_norm, ndwi_list_norm = [], [], []

    for idx in range(len(bands_patches[list(bands_patches.keys())[0]])):

        vv = bands_patches['VV'][idx]
        vh = bands_patches['VH'][idx]
        nir = bands_patches['B8'][idx]
        red = bands_patches['B4'][idx]
        swir1 = bands_patches['B11'][idx]

        cr = np.nan_to_num(vh/vv)
        cr_list.append(cr)
        ndvi = np.nan_to_num((nir-red)/(nir+red))
        ndvi_list.append(ndvi) 
        ndwi = np.nan_to_num((nir-swir1)/(nir+swir1))
        ndwi_list.append(ndwi)

    [cr_list_norm.append((data-np.min(data))/(np.max(data)-np.min(data))) for data in cr_list] 
    [ndvi_list_norm.append((data-np.min(data))/(np.max(data)-np.min(data))) for data in ndvi_list]
    [ndwi_list_norm.append((data-np.min(data))/(np.max(data)-np.min(data))) for data in ndwi_list]    

    bands_patches["CR"] = cr_list_norm
    bands_patches["NDVI"] = ndvi_list_norm
    bands_patches["NDWI"] = ndwi_list_norm

    return bands_patches

def calculateIndizesSen2(bands_patches):

    ndvi_list, ndwi_list = [], []
    ndvi_list_norm, ndwi_list_norm = [], []

    for idx in range(len(bands_patches[list(bands_patches.keys())[0]])):

        nir = bands_patches['B8'][idx]
        red = bands_patches['B4'][idx]
        swir1 = bands_patches['B11'][idx]

        ndvi = np.nan_to_num((nir-red)/(nir+red))
        ndvi_list.append(ndvi) 
        ndwi = np.nan_to_num((nir-swir1)/(nir+swir1))
        ndwi_list.append(ndwi)

    [ndvi_list_norm.append((data-np.min(data))/(np.max(data)-np.min(data))) for data in ndvi_list]
    [ndwi_list_norm.append((data-np.min(data))/(np.max(data)-np.min(data))) for data in ndwi_list]   

    bands_patches["NDVI"] = ndvi_list_norm
    bands_patches["NDWI"] = ndwi_list_norm

    print("Calculated NDVI")
    print("Calculated NDWI")

    return bands_patches

def imageAugmentation(images_path, masks_path, seed):

    def rotation90(image, seed):
        random.seed(seed)
        r_image = np.rot90(image)
        return r_image

    def h_flip(image, seed):
        random.seed(seed)
        hflipped_img= np.fliplr(image)
        return hflipped_img

    def v_flip(image, seed):
        random.seed(seed)
        vflipped_img= np.flipud(image)
        return vflipped_img

    def v_transl(image, seed):
        random.seed(seed)
        n_pixels = random.randint(-image.shape[0],image.shape[1])
        vtranslated_img = np.roll(image, n_pixels, axis=0)
        return vtranslated_img

    def h_transl(image, seed):
        random.seed(seed)
        n_pixels = random.randint(-image.shape[0],image.shape[0])
        htranslated_img = np.roll(image, n_pixels, axis=1)
        return htranslated_img

    transformations = {'rotate': rotation90, 'horizontal flip': h_flip,'vertical flip': v_flip, 'vertical shift': v_transl, 'horizontal shift': h_transl}         

    images=[] 
    masks=[]

    for im in os.listdir(images_path):      
        images.append(os.path.join(images_path,im))

    for msk in os.listdir(masks_path):  
        masks.append(os.path.join(masks_path,msk))
    
    images.sort()
    masks.sort()      

    for i in range(len(masks)): 
        
        image = images[i]
        mask = masks[i]

        original_image = load_img_as_array(image)
        original_mask = load_img_as_array(mask)
        
        for idx, transformation in enumerate(list(transformations)): 

            transformed_image = transformations[transformation](original_image, seed)
            transformed_mask = transformations[transformation](original_mask, seed)

            new_image_path= image.split(".")[0] + "_aug{}.tif".format(idx)
            new_mask_path = mask.split(".")[0] + "_aug{}.tif".format(idx)
            
            new_img = rasterio.open(new_image_path,'w', driver='Gtiff',
                        width=transformed_image.shape[0], height=transformed_image.shape[1],
                        count=original_image.shape[-1],
                        dtype=rasterio.float64)
            
            for band in range(transformed_image.shape[-1]-1):
                new_img.write(transformed_image[:,:,band], band+1)
            new_img.close() 
            
            tiff.imwrite(new_mask_path, transformed_mask)
            
            # if i == 25: 
            #     rows, cols = 2, 2
            #     plt.figure(figsize=(12,12))
        
            #     plt.subplot(rows, cols, 1)
            #     plt.imshow(original_image[:,:,:3])
            #     plt.subplot(rows, cols, 2)
            #     plt.imshow(transformed_image[:,:,:3])
            #     plt.subplot(rows, cols, 3)
            #     plt.imshow(original_mask)
            #     plt.subplot(rows, cols, 4)
            #     plt.imshow(transformed_mask)
            #     plt.show()
   
    return

def ScenceFinderAOI(shape_path, satellite, processing_level, product_type, start_date, end_date, cloud_cover, output_format='json', maxRecords = 15):

    start_date = start_date.replace(":", "%3A") 
    end_date = end_date.replace(":", "%3A") 

    shape = gpd.read_file(shape_path)
    shape_4326 = shape.to_crs(epsg=4326)
    
    #wkt_list = list(shape_4326['geometry'][0].exterior.coords)
    # for idx, point in enumerate(wkt_list):
    #     s = s + str(point[0]) + "+" + str(point[1])
    #     if idx != len(wkt_list)-1:
    #         s = s + "%2C"
    #     else:
    #         pass
    shape_wkt = shape_4326.to_wkt()
    wkt_list = list(shape_wkt['geometry'])

    list_path = []

    for point in wkt_list:

        geometry = point    

        base_url = "http://finder.code-de.org/resto/api/collections/"

        if satellite == "Sentinel1":
            modified_url = f"{satellite}/search.{output_format}?{maxRecords}&startDate={start_date}&completionDate={end_date}&location=all&processingLevel={processing_level}&productType={product_type}&sortParam=startDate&sortOrder=descending&status=all&geometry={geometry}&dataset=ESA-DATASET"
        else:
            modified_url = f"{satellite}/search.{output_format}?{maxRecords}&startDate={start_date}&completionDate={end_date}&cloudCover=[0%2C{cloud_cover}]&location=all&processingLevel={processing_level}&productType={product_type}&sortParam=startDate&sortOrder=descending&status=all&geometry={geometry}&dataset=ESA-DATASET"

        url = base_url + modified_url

        resp = requests.get(url).json()

        for feature in resp['features']:
            list_path.append(feature['properties']['productIdentifier'])

    return list_path

def cutString(string):
    string = '_'.join(string.split("_")[-2:])
    return string

def filterSen12(sceneList, filterDate=True, filterID=True):
    sceneList = random.sample(sceneList, len(sceneList))
    final_list = []

    for item in sceneList:
        date = item[0].split("_")[-2]
        id = item[0].split("_")[-1]
        if len(final_list) == 0:
            final_list.append(item)
        else:
            count = 0
            for i in final_list:
                if filterDate:
                    if date in i[0]:
                        count += 1
                elif filterID:
                    if id in i[0]:
                        count += 1
            if count == 0:
                final_list.append(item)
                
    return final_list

def filterSen2(sceneList, filterDate=True, filterID=True):
    sceneList = random.sample(sceneList, len(sceneList))
    final_list = []

    for item in sceneList:
        date = item.split("_")[-2]
        id = item.split("_")[-1]
        if len(final_list) == 0:
            final_list.append(item)
        else:
            count = 0
            for i in final_list:
                if filterDate:
                    if date in i:
                        count += 1
                elif filterID:
                    if id in i:
                        count += 1
            if count == 0:
                final_list.append(item)
                
    return final_list
