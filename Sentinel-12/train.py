import sys
sys.path.append("/home/hoehn/code/scripts")

import os
from glob import glob
import yaml
from tools_model import dice_coef, dice_loss, load_img_as_array, load_trainData
from sklearn.model_selection import train_test_split
from unet import binary_unet, build_unet
from matplotlib import pyplot as plt
from datagen import CustomImageGeneratorTrain, CustomImageGeneratorTest
import tensorflow as tf
import numpy as np
import random 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        
        patch_size = data['model_parameter']['patch_size']
        loss_function = data['model_parameter']['loss_function']
        epochs = data['model_parameter']['epochs']
        indizes = data["indizes"]
        output_folder = data["output_folder"]

        seed = data["seed"]

# Use patches as trainings data for model
model_name = "Ad_bc_5_idx"

model_path = os.path.join(output_folder, "models", model_name)

# Load training + test data from local folder and sort paths
X_train, X_test, y_train, y_test = load_trainData(output_folder, indizes)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.10, shuffle=True, random_state = seed)

import pdb 
pdb.set_trace()

X_train.sort()
X_test.sort()
y_train.sort()
y_test.sort()

# Get parameter for ImageGenerator 
patch_array = load_img_as_array(X_train[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

# Load images and masks with an custom data generator
train_datagen = CustomImageGeneratorTrain(X_train, y_train, patch_xy, b_count)
val_datagen = CustomImageGeneratorTrain(X_val, y_val, patch_xy, b_count)
test_datagen = CustomImageGeneratorTest(X_test, y_test, patch_xy, b_count)

print("Size of train data: ", len(train_datagen))
print("Size of val data: ", len(val_datagen))
print("Size of test data: ", len(test_datagen))

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

# #Load model
model = binary_unet(patch_xy[0], patch_xy[1], b_count)  
# model = build_unet(patch_xy[0], patch_xy[1], b_count)

# metrics 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss_function, metrics=["accuracy",
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.BinaryIoU(name="iou")])

# Model fit 
#callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss")
#log_dir = os.path.join(output_folder, "models", "logs", model_name) 
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
checkpoint_path = os.path.join(output_folder, "models", "checkpoints", model_name)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_recall", mode='max', verbose=1, save_best_only=True, save_weights_only=False)

model.fit(train_datagen, validation_data=val_datagen, verbose=1, epochs=epochs)

print("test model first time")
model.evaluate(test_datagen)

# Save model for prediction
# model.load_weights(checkpoint_path)
# model.save(model_path)
# print("test model second time")
# model.evaluate(test_datagen)

import pdb
pdb.set_trace()

# Predict test files
# model = tf.keras.models.load_model(model_path, compile=True, custom_objects={'dice_coef': dice_coef})
model = tf.keras.models.load_model(model_path, compile=True)

pred_test = model.predict(test_datagen) # f.eg.(288,128,128,1)
pred_test = (pred_test > 0.5).astype(np.uint8) 

batch_size = test_datagen[0][1].shape[0]
batch_nr = 5

# test_datagen[0] = ((16,128,128,12),(16,128,128,1)) -> tupel of length 2
for i in range(batch_size): 
    
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(test_datagen[batch_nr][1][i]) 
    plt.subplot(122)
    j = i+(batch_nr * batch_size)
    plt.imshow(pred_test[j])
    plt.savefig("/home/hoehn/data/prediction/prediction{}.png".format(i)) 