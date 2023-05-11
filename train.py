import os
from glob import glob
import yaml
from tools_model import load_img_as_array, load_trainData, dice_metric, find_augFiles2, removeChars
from sklearn.model_selection import train_test_split
from unet import binary_unet, build_unet
from matplotlib import pyplot as plt
from datagen import CustomImageGeneratorTrain, CustomImageGeneratorTest
import tensorflow as tf
import random 
import json

# Allocate only 80% of GPU memory
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1-0.2)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        
        patch_size = data['model_parameter']['patch_size']
        epochs = data['model_parameter']['epochs']
        indizes = data["indizes"]
        output_folder = data["output_folder"]
        val_metric = data["model_parameter"]["val_metric"]
        val_metric = "val_" + str(val_metric)
        seed = data["seed"]
        model_name = data["model_parameter"]["name"]

model_dir = os.path.join(output_folder, "models", model_name)
print("Training model in folder: ", model_dir)

# If model exist with same parameter delete this model      
if not os.path.exists(model_dir):
    # Create new folder structure
    os.mkdir(model_dir)
    os.mkdir(os.path.join(model_dir, "checkpoints"))
    os.mkdir(os.path.join(model_dir, "logs"))

# Load training + test data from local folder and sort paths after preprocessing 
X_train, X_test, y_train, y_test, base_folder = load_trainData(output_folder, indizes, patch_size)
print(f"Training dataset: {len(X_train)} Test dataset: {len(X_test)}")

# Load image parameter for image generator 
patch_array = load_img_as_array(X_train[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

augImg_folder = os.path.join(base_folder, "train/img_aug")
augMask_folder = os.path.join(base_folder, "train/mask_aug")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

print("-------------------- Trainig dataset --------------------")
X_train_aug, y_train_aug = find_augFiles2(X_train,y_train, augImg_folder, augMask_folder, seed)
# print("-------------------- Validation dataset --------------------")
# X_val_aug,  y_val_aug  = find_augFiles(X_val, y_val, augImg_folder, augMask_folder)
print("-------------------- Validation dataset -----------------")
X_val.sort()
y_val.sort() 
# check if dataset have equal order 
temp_X = list(map(removeChars, X_val))
temp_y = list(map(removeChars, y_val))   
print("X_val length after adding augementation files: ", len(X_val))
print("y_val length after adding augmentation files: ", len(y_val))
print("Does X_val and y_val have a equal size of files?:{}".format(temp_X==temp_y))
print("Does X_val and y_val have the same structure?:{}".format(len(temp_X)==len(temp_y)))
print("---------------------------------------------------------")

train_datagen = CustomImageGeneratorTrain(X_train_aug, y_train_aug, patch_xy, b_count)
#val_datagen = CustomImageGeneratorTrain(X_val_aug, y_val_aug, patch_xy, b_count)
val_datagen = CustomImageGeneratorTrain(X_val, y_val, patch_xy, b_count)
test_datagen = CustomImageGeneratorTest(X_test, y_test, patch_xy, b_count)

# sanity check
batch_nr = 5 #random.randint(0, len(train_datagen))
X,y = train_datagen[batch_nr] # train_datagen[0] = ((16,128,128,12),(16,128,128,1)) -> tupel
sc_folder = "/".join(X_train[0].split("/")[:-3]) + "/sanityCheck"

for i in range(X.shape[0]):
        
    plt.figure(figsize=(12,6))
    plt.title("Example of training data sample for Sentinel-2", fontsize = 12)
    plt.subplot(121)
    plt.title("False-colour composite (BGR) of patch", fontsize = 12)
    plt.imshow(X[i][:,:,:]) # 0:B11 1:B12 2:B2 3:B3 4:B4 ... # VH VV 
    plt.subplot(122)
    plt.title("Binary Mask of patch", fontsize = 12)
    plt.imshow(y[i])
    plt.show()
    plt.savefig(os.path.join(output_folder, "{}/single_{}.png".format(sc_folder, i))) 

#Load model
model = binary_unet(patch_xy[0], patch_xy[1], b_count)  

# Model fit 
log_dir = os.path.join(model_dir, "logs", f"logs") 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
checkpoint_path = os.path.join(model_dir, "checkpoints", f"best_weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor=val_metric, mode='max', verbose=1, save_best_only=True, save_weights_only=True)
earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor=val_metric, mode="max", verbose=1, patience=15)

# metrics 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[
    "accuracy",
    dice_metric,
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.BinaryIoU(name="iou")])

model.fit(train_datagen, validation_data=val_datagen, verbose=1, epochs=epochs, callbacks=[checkpoint_callback, tensorboard_callback, earlyStop_callback])
model.load_weights(checkpoint_path)
model.save(os.path.join(model_dir, "best_model"))

print("Evaluate on validation data")
val_results = model.evaluate(val_datagen)
test_results = model.evaluate(test_datagen)
val_r_dict = dict(zip(model.metrics_names,val_results))
test_r_dict = dict(zip(model.metrics_names, test_results))

file = os.path.join(model_dir, "metrics.txt")
if os.path.exists(file):
    open(file, "w").close() # empty textfile for new input
else:
    with open(file, "w") as document: pass # create empty textfile

json.dump(val_r_dict, open(file, "w"))
json.dump(test_r_dict, open(file, "a"))

