import rasterio
from patchify import patchify,unpatchify
import numpy as np
import os
from matplotlib import pyplot as plt
from keras import backend as K
import json
import tifffile as tiff
import random
from glob import glob
import tensorflow as tf

def load_img_as_array(path):
    # read img as array 
    img_array = rasterio.open(path).read()
    img_array = np.moveaxis(img_array, 0, -1)
    # img_array = np.nan_to_num(img_array)
    return img_array

# def dice_metric(y_pred, y_true, smooth=1e-5):
#     intersection = tf.reduce_sum(y_pred * y_true)
#     union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
#     dice = (2* intersection + smooth) / union + smooth
#     dice = tf.reduce_mean(dice, name='dice_coe')
#     return dice

#Dice metric can be a great metric to track accuracy of semantic segmentation.
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2*intersection / union

def jaccard_metric(y_pred, y_true, smooth=1e-5):
    intersection = tf.reduce_sum(y_pred * y_true)
    union = tf.reduce_sum(y_pred*y_pred) + tf.reduce_sum(y_true*y_true)
    dice = (intersection + smooth) / (union + smooth)
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice

def predictPatches(model, predict_datagen, raster_path, output_folder):

    prediction = model.predict(predict_datagen, verbose=1) 
    prediction = (prediction > 0.5).astype(np.uint8) 
    
    raster = rasterio.open(raster_path)
    patch_size = prediction.shape[1]
    x = int(raster.width/patch_size)
    y = int(raster.height/patch_size)
    patches_shape = (x, y, 1, patch_size, patch_size, 1)

    prediciton_reshape = np.reshape(prediction, patches_shape) 

    recon_predict = unpatchify(prediciton_reshape, (patch_size*x,patch_size*y,1))

    transform = raster.transform
    crs = raster.crs
    name = os.path.basename(raster_path).split("_")[3]
    predict_out = os.path.join(output_folder, name + "_prediction.tif")

    final = rasterio.open(predict_out, mode='w', driver='Gtiff',
                    width=recon_predict.shape[0], height=recon_predict.shape[1],
                    count=1,
                    crs=crs,
                    transform=transform,
                    dtype=rasterio.int16)

    final.write(recon_predict[:,:,0],1) 
    final.close()
    print("Reconstructed and predicted sentinel tile and saved in: ", predict_out)
    
    return recon_predict

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

def createMetrics(file, metrics_dict, name_data):

    sum = {}
    append_new_line(file, f"--------------------- {name_data} ---------------------")
    for key, value in metrics_dict.items():
        append_new_line(file, f"--------------------- {str(key)} ---------------------")
        append_new_line(file, json.dumps(value))
        for k,v in value.items():
            if sum.get(k) is None:
                sum[k] = v
            else:
                sum[k] += v

    mean = {key: value/len(metrics_dict) for key, value in sum.items()}

    append_new_line(file, "--------------------- Mean metrics ---------------------")
    [append_new_line(file, f"{key}: {value}") for key, value in mean.items()]

def load_trainData(output_folder, idx):

    if idx:
        X_train = glob("{}/crops/idx/train/img/*.tif".format(output_folder))
        y_train = glob("{}/crops/idx/train/mask/*.tif".format(output_folder))
        X_test = glob("{}/crops/idx/test/img/*.tif".format(output_folder))
        y_test = glob("{}/crops/idx/test/mask/*.tif".format(output_folder))
    else:
        X_train = glob("{}/crops/no_idx/train/img/*.tif".format(output_folder))
        y_train = glob("{}/crops/no_idx/train/mask/*.tif".format(output_folder))
        X_test = glob("{}/crops/no_idx/test/img/*.tif".format(output_folder))
        y_test = glob("{}/crops/no_idx/test/mask/*.tif".format(output_folder))

    X_train.sort()
    X_test.sort()
    y_train.sort()
    y_test.sort()

    return X_train, X_test, y_train, y_test

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

def find_augFiles(X_train, y_train, dir_augImg, dir_augMask):

    def removeChars(input_string):
        string_split = os.path.basename(input_string).split("_")
        string_split.pop(1)
        new_string = "_".join(string_split)
        return new_string
       
    X_train_aug = []
    y_train_aug = []

    X_train.sort()
    y_train.sort()

    print("X_train length before adding augementation files: ", len(X_train))
    print("y_train length before adding augmentation files: ", len(y_train))

    for img in X_train:
        img_name = os.path.basename(img).split(".")[0]
        for i in range(3):
            aug_img_path = os.path.join(dir_augImg, img_name + f"_aug{i}.tif" )
            X_train_aug.append(aug_img_path)

    for mask in y_train:
        mask_name = os.path.basename(mask).split(".")[0]
        for i in range(3):
            aug_mask_path = os.path.join(dir_augMask, mask_name + f"_aug{i}.tif" )
            y_train_aug.append(aug_mask_path)

    X_train_aug += X_train
    y_train_aug += y_train

    X_train_aug.sort()
    y_train_aug.sort()

    print("X_train length after adding augementation files: ", len(X_train_aug))
    print("y_train length after adding augmentation files: ", len(y_train_aug))

    # check if dataset have equal order 
    temp_X = list(map(removeChars, X_train_aug))
    temp_y = list(map(removeChars, y_train_aug))
    
    print("Does X_train and y_train have a equal size of files?:{}".format(temp_X==temp_y))
    print("Does X_train and y_train have the same structure?:{}".format(len(temp_X)==len(temp_y)))
    
    temp_Xy = list(zip(X_train_aug, y_train_aug))
    random.shuffle(temp_Xy)
    X_train_aug, y_train_aug = zip(*temp_Xy) 

    return X_train_aug, y_train_aug 
    