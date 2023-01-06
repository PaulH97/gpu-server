import os
from glob import glob
import yaml
from tools_model import dice_coef, dice_loss, load_img_as_array, load_trainData
from sklearn.model_selection import train_test_split
from unet import binary_unet
from matplotlib import pyplot as plt
from datagen import CustomImageGenerator
import tensorflow as tf
import numpy as np
import random
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        
        patch_size = data['model_parameter']['patch_size']
        optimizer = data['model_parameter']['optimizer']
        loss_function = data['model_parameter']['loss_function']
        epochs = data['model_parameter']['epochs']
        indizes = data["indizes"]
        output_folder = data["output_folder"]

        seed = data["seed"]

# Use patches as trainings data for model
parameter = {"Adam": "Ad", "binary_crossentropy": "bc"}

if indizes: 
    model_name = parameter[optimizer] + "_" + parameter[loss_function] + "_" + str(epochs) + "_idx" # Ad_bc_100_idx 
else:
    model_name = parameter[optimizer] + "_" + parameter[loss_function] + "_" + str(epochs) # Ad_bc_100

model_path = os.path.join(output_folder, "models", model_name)

# Load training + test data from local folder and sort paths
X_train, X_test, y_train, y_test = load_trainData(output_folder, indizes)

X_train.sort()
X_test.sort()
y_train.sort()
y_test.sort()

# Get parameter for ImageGenerator 
patch_array = load_img_as_array(X_train[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

# Load images and masks with an custom data generator
train_datagen = CustomImageGenerator(X_train, y_train, patch_xy, b_count)
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
model.compile(optimizer=optimizer, loss=loss_function, metrics=[dice_coef,
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.BinaryIoU(name="iou")])

# Model fit 
#callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss")
#log_dir = os.path.join(output_folder, "models", "logs", model_name) 
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
#checkpoint_path = os.path.join(output_folder, "models", "checkpoints", model_name)
#checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="iou", mode='max', verbose=1, save_best_only=True, save_weights_only=True)

model.fit(train_datagen, verbose=1, epochs=epochs)

# Save model for prediction
model.save(model_path)

#model.load_weights(checkpoint_path)
model.evaluate(test_datagen)

# pred_test = model.predict(test_datagen) # f.eg.(288,128,128,1)
# pred_test = (pred_test > 0.5).astype(np.uint8) 

# for i in range((test_datagen[1][1].shape[0])):

#     plt.figure(figsize=(12,6))
#     plt.subplot(121)
#     plt.imshow(test_datagen[0][1][i])
#     plt.subplot(122)
#     plt.imshow(pred_test[i])
#     plt.show()
#     plt.savefig("/home/hoehn/data/prediction/prediction{}.png".format(i)) 