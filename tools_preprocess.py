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
import shutil
from glob import glob

np.seterr(divide='ignore', invalid='ignore')

def getTileBandsRaster(tile_folder, model_id, idx_dir):
        
    if model_id == "Sentinel-12":   
        # get tile name and path for each band
        tile_name = tile_folder[0].split("_")[-1]
        sen_path =  glob(f"{tile_folder[0]}/*.tif") + glob(f"{tile_folder[1]}/*.tif") + glob(f"{idx_dir}/{tile_name}*.tif")
        sen_path.sort() # VH VV B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A 
    
    else:
        # get tile name and path for each band
        tile_folder = tile_folder.strip() # remove \n from file path
        tile_name = tile_folder.split("_")[-1] 
        if model_id == "Sentinel-2":
            sen_path = glob(f"{tile_folder}/*.tif") + [f"{idx_dir}/{tile_name}_ndvi.tif", f"{idx_dir}/{tile_name}_ndwi.tif"]
        else:
            sen_path = glob(f"{tile_folder}/*.tif") + [f"{idx_dir}/{tile_name}_cr.tif"]
        sen_path.sort() # VH VV or B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A 
        
    raster_muster = [i for i in sen_path if i.find("B2") > 0 or i.find("VH") > 0][-1]

    return tile_name, sen_path, raster_muster

def getBandPaths(tile_folder, model_id):

    if model_id == "Sentinel-12":   
        # get tile name and path for each band
        sen_path =  glob(f"{tile_folder[0]}/*.tif") + glob(f"{tile_folder[1]}/*.tif")
        sen_path.sort() #  VH VV B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A 
    else:
        # remove \n from file path
        tile_folder = tile_folder.strip()
        sen_path = glob(f"{tile_folder}/*.tif") 
        sen_path.sort() # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A or VH VV

    return sen_path

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
    #output_folder = os.path.join(output_folder, "masks")
    r_out = os.path.join(output_folder, os.path.basename(vector_path).split(".")[0] + "_" + tile_name +".tif")

    with rasterio.open(r_out, 'w+', driver='GTiff',
            height = raster.height, width = raster.width,
            count = 1, dtype="int16", crs = crs, transform=transform) as rst:
        out_arr = rst.read(1)
        rasterized = features.rasterize(shapes=geom_value, fill=0, out=out_arr, transform = rst.transform)
        rst.write_band(1, rasterized)
    rst.close()
    return r_out

def patchify_band(array, patch_size):

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

def savePatchesPV(patches, img_out, mask_out, seed, raster_muster):
    
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
                final.set_band_description(band_nr+1, band_name)
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
                final.set_band_description(band_nr+1, band_name)
            final.close()
        
    return 

def savePatchesFullImg(patches, output_folder):
    
    band_names = list(patches.keys())
    band_names.sort()

    for idx in range(len(patches[band_names[0]])):

        final = rasterio.open(os.path.join(output_folder, f'img_{idx}.tif'),'w', driver='Gtiff',
                        width=patches[band_names[0]][0].shape[0], height=patches[band_names[0]][0].shape[1],
                        count=len(band_names),
                        dtype=rasterio.float64)

        for band_nr, band_name in enumerate(band_names):
            final.write(patches[band_name][idx][:,:,0],band_nr+1)
            final.set_band_description(band_nr+1, band_name)
        final.close()

    return 

def calculate_indices(sen_paths, tile_id, bands_dict, out_dir, satellite_type):
    red, nir, swir1, vv, vh = None, None, None, None, None

    if satellite_type in ["Sentinel-12", "Sentinel-2"]:
        red = load_img_as_array([i for i in sen_paths if i.find("B4") > 0][0])
        nir = load_img_as_array([i for i in sen_paths if i.find("B8") > 0][0])
        swir1_ds = resampleRaster([i for i in sen_paths if i.find("B11") > 0][0], 10)
        swir1 = swir1_ds.ReadAsArray()
        swir1 = np.expand_dims(swir1, axis=0)
        swir1 = np.moveaxis(swir1, 0, -1)
        swir1 = np.nan_to_num(swir1)

    if satellite_type in ["Sentinel-12", "Sentinel-1"]:
        vv = load_img_as_array([i for i in sen_paths if i.find("VV") > 0][0])
        vh = load_img_as_array([i for i in sen_paths if i.find("VH") > 0][0])

    if satellite_type == "Sentinel-12":
        cr = np.nan_to_num(vh - vv)
        bands_dict["CR"].append(cr)
        tiff.imwrite(os.path.join(out_dir, f'{tile_id}_cr.tif'), cr)
        print("Calculated CR")

    if satellite_type in ["Sentinel-12", "Sentinel-2"]:
        ndvi = np.nan_to_num((nir - red) / (nir + red))
        ndwi = np.nan_to_num((nir - swir1) / (nir + swir1))
        bands_dict["NDVI"].append(ndvi)
        bands_dict["NDWI"].append(ndwi)
        tiff.imwrite(os.path.join(out_dir, f'{tile_id}_ndvi.tif'), ndvi)
        tiff.imwrite(os.path.join(out_dir, f'{tile_id}_ndwi.tif'), ndwi)
        print("Calculated NDVI")
        print("Calculated NDWI")

    return out_dir

def calculateIndizesSen12(sen_paths, tile_id, bands_dict, out_dir):
        
    vv = load_img_as_array([i for i in sen_paths if i.find("VV")> 0][0])
    vh = load_img_as_array([i for i in sen_paths if i.find("VH")> 0][0])
    red = load_img_as_array([i for i in sen_paths if i.find("B4")> 0][0])
    nir = load_img_as_array([i for i in sen_paths if i.find("B8")> 0][0])
    swir1_ds = resampleRaster([i for i in sen_paths if i.find("B11")> 0][0], 10)
    swir1 = swir1_ds.ReadAsArray()
    swir1 = np.expand_dims(swir1, axis=0)
    swir1 = np.moveaxis(swir1, 0, -1)
    swir1 = np.nan_to_num(swir1)
        
    cr = np.nan_to_num(vh-vv)
    ndvi = np.nan_to_num((nir-red)/(nir+red))
    ndwi = np.nan_to_num((nir-swir1)/(nir+swir1))

    tiff.imwrite(os.path.join(out_dir, f'{tile_id}_cr.tif'), cr)
    tiff.imwrite(os.path.join(out_dir, f'{tile_id}_ndvi.tif'), ndvi)
    tiff.imwrite(os.path.join(out_dir, f'{tile_id}_ndwi.tif'), ndwi)

    bands_dict["CR"].append(cr)
    bands_dict["NDVI"].append(ndvi)
    bands_dict["NDWI"].append(ndwi)
    
    print("Calculated CR")
    print("Calculated NDVI")
    print("Calculated NDWI")

    return out_dir

def calculateIndizesSen2(sen_paths, tile_id, bands_dict, out_dir):
        
    red = load_img_as_array([i for i in sen_paths if i.find("B4")> 0][0])
    nir = load_img_as_array([i for i in sen_paths if i.find("B8")> 0][0])
    swir1_ds = resampleRaster([i for i in sen_paths if i.find("B11")> 0][0], 10)
    swir1 = swir1_ds.ReadAsArray()
    swir1 = np.expand_dims(swir1, axis=0)
    swir1 = np.moveaxis(swir1, 0, -1)
    swir1 = np.nan_to_num(swir1)
        
    ndvi = np.nan_to_num((nir-red)/(nir+red))
    ndwi = np.nan_to_num((nir-swir1)/(nir+swir1))

    tiff.imwrite(os.path.join(out_dir, f'{tile_id}_ndvi.tif'), ndvi)
    tiff.imwrite(os.path.join(out_dir, f'{tile_id}_ndwi.tif'), ndwi)
    
    bands_dict["NDVI"].append(ndvi)
    bands_dict["NDWI"].append(ndwi)
    
    print("Calculated NDVI")
    print("Calculated NDWI")

    return out_dir

def calculateIndizesSen1(sen_paths, tile_id, bands_dict, out_dir):
    
    vv = load_img_as_array([i for i in sen_paths if i.find("VV")> 0][0])
    vh = load_img_as_array([i for i in sen_paths if i.find("VH")> 0][0])
    cr = np.nan_to_num(vh-vv)
    
    tiff.imwrite(os.path.join(out_dir, f'{tile_id}_cr.tif'), cr)
    
    bands_dict["CR"].append(cr)
    
    print("Calculated CR")

    return out_dir

def imageAugmentation2(images_path, masks_path, seed):

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

    images_path.sort()
    masks_path.sort()     

    for i in range(len(masks_path)): 
        
        image = images_path[i]
        mask = masks_path[i]

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

def imageAugmentation(X_train, y_train, seed):

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

    #transformations = {'rotate': rotation90, 'horizontal flip': h_flip,'vertical flip': v_flip, 'vertical shift': v_transl, 'horizontal shift': h_transl}         
    transformations = {'rotate': rotation90, 'horizontal flip': h_flip,'vertical flip': v_flip}         

    # Create folder for augmented images - so that they got not mixed up with the original images
    augImg_folder = os.path.join("/".join(X_train[0].split("/")[:-2]), "img_aug")
    augMask_folder = os.path.join("/".join(y_train[0].split("/")[:-2]), "mask_aug")

    if not os.path.isdir(augImg_folder):
        os.makedirs(augImg_folder)
    else:
        for f in os.listdir(augImg_folder):
            os.remove(os.path.join(augImg_folder, f))
    
    if not os.path.isdir(augMask_folder):
        os.makedirs(augMask_folder)
    else:
        for f in os.listdir(augMask_folder):
            os.remove(os.path.join(augMask_folder, f))
    
    X_train.sort()
    y_train.sort()    

    for i in range(len(y_train)): 
        
        image = X_train[i]
        mask = y_train[i]

        original_image = load_img_as_array(image)
        original_mask = load_img_as_array(mask)
        
        for idx, transformation in enumerate(list(transformations)): 

            transformed_image = transformations[transformation](original_image, seed)
            transformed_mask = transformations[transformation](original_mask, seed)

            new_img_name = image.split("/")[-1].split(".")[0] + "_aug{}.tif".format(idx)
            new_mask_name = mask.split("/")[-1].split(".")[0] + "_aug{}.tif".format(idx)
            
            new_image_path= os.path.join(augImg_folder, new_img_name)
            new_mask_path = os.path.join(augMask_folder, new_mask_name)

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
        
    return augImg_folder, augMask_folder 

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

def filterSen1(sceneList, filterDate=True, filterID=True):
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

def rebuildCropFolder(crop_folder):

    train = os.path.join(crop_folder, "train")
    test = os.path.join(crop_folder, "test")
    prediction = os.path.join(crop_folder, "prediction")
    sanity_check = os.path.join(crop_folder, "sanityCheck")
    train_img = os.path.join(train, "img")
    train_mask = os.path.join(train, "mask")
    test_img = os.path.join(test, "img")
    test_mask = os.path.join(test, "mask")
    pred_full = os.path.join(prediction, "full_img")
    pred_img = os.path.join(prediction, "img")
    pred_mask = os.path.join(prediction, "mask")

    if os.path.exists(crop_folder):
        shutil.rmtree(crop_folder)
    
    os.mkdir(crop_folder)
    os.mkdir(train)
    os.mkdir(test)
    os.mkdir(train_img)
    os.mkdir(train_mask)
    os.mkdir(test_img)
    os.mkdir(test_mask)
    os.mkdir(prediction)
    os.mkdir(pred_full)
    os.mkdir(pred_img)
    os.mkdir(pred_mask)
    os.mkdir(sanity_check)

    return train_img, train_mask, pred_img, pred_mask, pred_full