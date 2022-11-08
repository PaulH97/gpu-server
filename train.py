import os
from glob import glob
import yaml
from tools_train_predict import dice_coef, load_img_as_array
from sklearn.model_selection import train_test_split
from unet import binary_unet
import random
from matplotlib import pyplot as plt
from datagen import CustomImageGenerator
import tensorflow as tf
from datetime import datetime
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Read data from config file
if os.path.exists("config_training.yaml"):
    with open('config_training.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        
        patch_size = data['model']['patch_size']
        optimizer = data['model']['optimizer']
        loss_function = data['model']['loss_function']
        epochs = data['model']['epochs']

        output_folder = data["output_folder"]

# Use patches as trainings data for model
img_list = glob("{}/Sen12_indizes_crops/img/*.tif".format(output_folder))
mask_list = glob("{}/Sen12_indizes_crops/mask/*.tif".format(output_folder))

model_name = "/home/hoehn/code/Sen12_indizes" + optimizer + loss_function + str(epochs) + "_" + datetime.now().strftime("%Y%m%d")

img_list.sort()
mask_list.sort()

# Split training data
X_train, X_test, y_train, y_test = train_test_split(img_list, mask_list, test_size = 0.20, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.10, random_state = 42)

# Load images and masks with an custom data generator - for performance reason
patch_array = load_img_as_array(X_train[0])

import pdb
pdb.set_trace()

patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

train_datagen = CustomImageGenerator(X_train, y_train, patch_xy, b_count)
val_datagen = CustomImageGenerator(X_val,y_val, patch_xy, b_count)
test_datagen = CustomImageGenerator(X_test, y_test, patch_xy, b_count)

# sanity check
# batch_nr = random.randint(0, len(train_datagen))
# X,y = train_datagen[batch_nr]

# for i in range(X.shape[0]):

#     plt.figure(figsize=(12,6))
#     plt.subplot(121)
#     plt.imshow(X[i][:,:,4])
#     plt.subplot(122)
#     plt.imshow(y[i])
#     plt.show()
#     plt.savefig("sanity_check{}.png".format(i)) 
                                                                                                                                                                                                                                                                                                                                                        
#Load model
model = binary_unet(patch_xy[0], patch_xy[1], b_count)  

# metrics 
model.compile(optimizer=optimizer, loss=loss_function, metrics=[dice_coef])

# Model fit 
#callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss")
logdir = "logs/" 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

model.fit(train_datagen, validation_data=val_datagen, verbose=1, epochs=epochs, callbacks=[tensorboard_callback])

# Save model for prediction
model.save(model_name)

pred_test = model.predict(test_datagen) # f.eg.(288,128,128,1)
pred_test = (pred_test > 0.5).astype(np.uint8) 

for i in range((test_datagen[0][1].shape[0])):

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(test_datagen[0][1][i])
    plt.subplot(122)
    plt.imshow(pred_test[i])
    plt.show()
    plt.savefig("prediction{}.png".format(i)) 


