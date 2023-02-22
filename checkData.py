import os
from tools_preprocess import load_img_as_array
import numpy as np
import rasterio

dir_train_img = "/home/hoehn/data/output/Sentinel-12/crops256/idx/train/img_aug"
dir_train_mask = "/home/hoehn/data/output/Sentinel-12/crops256/idx/train/mask_aug"

img_count = len(os.listdir(dir_train_img))
mask_count = len(os.listdir(dir_train_mask))

print("Is the length of image folder same as for the mask folder?")
print("{}={}: {}".format(img_count, mask_count,(img_count==mask_count)))

equal = []
missing = []

for i in os.listdir(dir_train_img):
    i = "_".join(["mask" if j == "img" else j for j in i.split("_")]) 

    if i not in os.listdir(dir_train_mask):
        missing.append(i)
    else:
        equal.append(i)

print("Found {} equal files in both dirs".format(len(equal)))
print("Found {} missing files in both dirs".format(len(missing)))
print(missing)
#
# dirpath_test_img = "/home/hoehn/data/output/Sentinel-12/crops256/no_idx/train/img"
# dir_test_img = os.listdir(dirpath_test_img)
# dir_test_img_new = "/home/hoehn/data/output/Sentinel-12/crops256/idx/train/img"

# dir_test_img.sort()

# for img in dir_test_img:
#     img_path = os.path.join(dirpath_test_img, img)
#     array = load_img_as_array(img_path) # (256,256,12) -> 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'VH', 'VV'
   
#     vv = array[:,:,11]
#     vh = array[:,:,10]
#     nir = array[:,:,8]
#     red = array[:,:,4]
#     swir1 = array[:,:,0]
        
#     cr = np.nan_to_num(vh/vv)
#     ndvi = np.nan_to_num((nir-red)/(nir+red))
#     ndwi = np.nan_to_num((nir-swir1)/(nir+swir1))
    
#     cr = (cr-np.min(cr))/(np.max(cr)-np.min(cr))
#     ndvi = (ndvi-np.min(ndvi))/(np.max(ndvi)-np.min(ndvi))
#     ndwi = (ndwi-np.min(ndwi))/(np.max(ndwi)-np.min(ndwi))   

#     array_new = np.dstack((array, cr, ndvi, ndwi))
        
#     new_image_path= os.path.join(dir_test_img_new, img)
                
#     new_img = rasterio.open(new_image_path,'w', driver='Gtiff',
#                 width=array.shape[0], height=array.shape[1],
#                 count=array_new.shape[-1],
#                 dtype=rasterio.float64)
    
#     new_img.write(array_new[:,:,0], 1)
#     new_img.write(array_new[:,:,1], 2)
#     new_img.write(array_new[:,:,2], 3)
#     new_img.write(array_new[:,:,3], 4)
#     new_img.write(array_new[:,:,4], 5)
#     new_img.write(array_new[:,:,5], 6)
#     new_img.write(array_new[:,:,6], 7)
#     new_img.write(array_new[:,:,7], 8)
#     new_img.write(array_new[:,:,8], 9)
#     new_img.write(array_new[:,:,9], 10)
#     new_img.write(array_new[:,:,12],11)
#     new_img.write(array_new[:,:,13],12)
#     new_img.write(array_new[:,:,14],13)
#     new_img.write(array_new[:,:,10],14)
#     new_img.write(array_new[:,:,11],15)
#     new_img.close() 
