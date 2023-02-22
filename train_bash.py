import os
from glob import glob
import yaml
from tools_model import load_img_as_array, createMetrics, load_trainData, find_augFiles2, removeChars, dice_metric
from sklearn.model_selection import KFold
from unet import binary_unet
from matplotlib import pyplot as plt
from datagen import CustomImageGeneratorTrain, CustomImageGeneratorTest
import tensorflow as tf
import random 
import json
import argparse

# Allocate only 80% of GPU memory
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1-0.15)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Definieren der Argumente
parser = argparse.ArgumentParser()
parser.add_argument('--run_number', type=int, required=True, help='Current run number')
args = parser.parse_args()

# Zugriff auf die Argumente
run_number = args.run_number

# Ausgabe der aktuellen Nummer
model_ID = "model_" + str(run_number)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Read data from config file
if os.path.exists("config_train.yaml"):
    with open('config_train.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        
        output_folder = data["model_params"][model_ID]["folder"]
        patch_size = data["model_params"][model_ID]["patch_size"]
        indizes = data["model_params"][model_ID]["idx"]
        model_name = data["model_params"][model_ID]["name"]
        epochs = data["training_params"]["epochs"]
        k_folds = data["training_params"]["kfold"]
        seed = data["training_params"]["seed"]
        
model_dir = os.path.join(output_folder, "models2", model_name)
print("Training model in folder: ", model_dir)

# If model exist with same parameter delete this model      
if not os.path.exists(model_dir):
    # Create new folder structure
    os.mkdir(model_dir)
    os.mkdir(os.path.join(model_dir, "checkpoints"))
    os.mkdir(os.path.join(model_dir, "logs"))

# Load training + test data from local folder and sort paths after preprocessing 
X_train, X_test, y_train, y_test, base_folder = load_trainData(output_folder, indizes, patch_size)
X_train += X_test 
y_train += y_test

# Load image parameter for image generator 
patch_array = load_img_as_array(X_train[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

augImg_folder = os.path.join(base_folder, "train/img_aug")
augMask_folder = os.path.join(base_folder, "train/mask_aug")

# Completly held out data as generator - for later evaluation
# test_datagen = CustomImageGeneratorTest(X_test, y_test, patch_xy, b_count)

## Training with K-fold cross validation
kf = KFold(n_splits=k_folds, random_state=seed, shuffle=True)
fold_var = 1

test_metrics = {}
val_metrics = {}

# validation data is just a different name for test data in kfold cross validation 
for train_index, val_index in kf.split(X_train):

    print("Start with kfold {}:".format(fold_var))

    # List of paths to images based on train_index - split data in kfold parts
    X_train_cv = [X_train[idx] for idx in train_index]
    y_train_cv = [y_train[idx] for idx in train_index]
    X_val_cv = [X_train[idx] for idx in val_index] 
    y_val_cv = [y_train[idx] for idx in val_index]
 
    print("-------------------- Trainig dataset --------------------")
    X_train_aug, y_train_aug = find_augFiles2(X_train_cv,y_train_cv, augImg_folder, augMask_folder, seed)
    print("---------------------------------------------------------")
    print("-------------------- Validation dataset -----------------")
    X_val_cv.sort()
    y_val_cv.sort() 
    # check if dataset have equal order 
    temp_X = list(map(removeChars, X_val_cv))
    temp_y = list(map(removeChars, y_val_cv))   
    print("X_val length after adding augementation files: ", len(X_val_cv))
    print("y_val length after adding augmentation files: ", len(y_val_cv))
    print("Does X_val and y_val have the same structure?:{}".format(temp_X==temp_y))
    print("Does X_val and y_val have a equal size of files?:{}".format(len(temp_X)==len(temp_y)))
    print("---------------------------------------------------------")
    
    print(f"Length of training dataset: {len(X_train_aug)}")

    while len(X_train_aug) % 2 != 0:
        X_train_aug = X_train_aug[:-1]
        y_train_aug = y_train_aug[:-1]

    print(f"Length of training dataset after slice: {len(X_train_aug)}")

    # Load images and masks with an custom data generator - for performance reason - can not load all data from disk
    train_datagen = CustomImageGeneratorTrain(X_train_aug, y_train_aug, patch_xy, b_count)
    val_datagen = CustomImageGeneratorTrain(X_val_cv, y_val_cv, patch_xy, b_count)

    # sanity check 
    batch_nr = 5
    X,y = train_datagen[batch_nr] # train_datagen[0] = ((16,128,128,12),(16,128,128,1)) -> tupel
    sc_folder = "/".join(X_train[0].split("/")[:-3]) + "/sanityCheck"
            
    for i in range(X.shape[0]):
        
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(X[i][:,:,2:5]) # 0:B11 1:B12 2:B2 3:B3 4:B4 ... # VH VV 
        plt.subplot(122)
        plt.imshow(y[i])
        plt.show()
        plt.savefig(os.path.join(output_folder, "{}/kfold_{}_{}.png".format(sc_folder,fold_var, i))) 
    
    print("Finished sanity check")
    
    #Load model
    model = binary_unet(patch_xy[0], patch_xy[1], b_count)  

    # Model fit 
    log_dir = os.path.join(model_dir, "logs", f"logs_k{fold_var}") 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint_path = os.path.join(model_dir, "checkpoints", f"best_weights_k{fold_var}.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_dice_metric", mode='max', verbose=1, save_best_only=True, save_weights_only=True)
    earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_dice_metric", mode="max", verbose=1, patience=15)

    # metrics 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[
        "accuracy",
        dice_metric,
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.BinaryIoU(name="iou")])

    model.fit(train_datagen, validation_data=val_datagen, verbose=1, epochs=epochs, callbacks=[checkpoint_callback, tensorboard_callback, earlyStop_callback])
    model.load_weights(checkpoint_path)
    
    print("Evaluate on validation data")
    val_results = model.evaluate(val_datagen)
    val_r_dict = dict(zip(model.metrics_names,val_results))
    val_metrics[f"Kfold_{fold_var}"] = val_r_dict 

    tf.keras.backend.clear_session()

    fold_var += 1

file = os.path.join(model_dir, "metrics.txt")
if os.path.exists(file):
    open(file, "w").close() # empty textfile for new input
else:
    with open(file, "w") as document: pass # create empty textfile

createMetrics(file, val_metrics, "Validation data")

# Find best model comparing one metric from all kfolds and save it - choose test or validation
list_metrics = []

for kfold, metric_dict in val_metrics.items():
    for metric_name, metric_value in metric_dict.items():
        if metric_name == "dice_metric":
            list_metrics.append(metric_value)

max_value = max(list_metrics)
max_index = list_metrics.index(max_value)
checkpoint_path = os.path.join(model_dir, "checkpoints", f"best_weights_k{max_index+1}.h5")
print("Best model weights of validation data: ", checkpoint_path)

model.load_weights(checkpoint_path)
model.save(os.path.join(model_dir, "best_model"))

print("Please insert the following path of the best model into the config.yaml file!: ", os.path.join(model_dir, "best_model") )