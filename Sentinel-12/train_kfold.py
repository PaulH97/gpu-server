import sys
sys.path.append("/home/hoehn/code/scripts")
import os
from glob import glob
import yaml
from tools_model import dice_coef, load_img_as_array, createMetrics, load_trainData, imageAugmentation, find_augFiles
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from unet import binary_unet
from matplotlib import pyplot as plt
from datagen import CustomImageGeneratorTrain, CustomImageGeneratorTest
import tensorflow as tf
import shutil
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
        k_folds = data['model_parameter']['kfold']
        seed = data["seed"]
        augImg_folder = data["augImg_folder"]
        augMask_folder = data["augMask_folder"]
        model_name = data['model_parameter']['name']
        model_dir = os.path.join(output_folder, "models", model_name)

# If model exist with same parameter delete this model      
if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)

# Create new folder structure
os.mkdir(model_dir)
os.mkdir(os.path.join(model_dir, "checkpoints"))
os.mkdir(os.path.join(model_dir, "logs"))

# Load training + test data from local folder and sort paths after preprocessing 
X_train, X_test, y_train, y_test = load_trainData(output_folder, indizes)

X_train.sort()
X_test.sort()
y_train.sort()
y_test.sort()

print(f"Training dataset: {len(X_train)} Test dataset: {len(X_test)}")

# Load images and masks with an custom data generator - for performance reason
patch_array = load_img_as_array(X_train[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

# Completly held out data as generator - for later evaluation
test_datagen = CustomImageGeneratorTest(X_test, y_test, patch_xy, b_count)

## Training with K-fold cross validation
kf = KFold(n_splits=k_folds, random_state=seed, shuffle=True)
fold_var = 1

test_metrics = {}
val_metrics = {}

# validation data is just a different name for test data in kfold cross validation 
for train_index, val_index in kf.split(X_train):

    # List of paths to images based on train_index - split data in kfold parts
    X_train_cv = [X_train[idx] for idx in train_index]
    y_train_cv = [y_train[idx] for idx in train_index]
    X_val_cv = [X_train[idx] for idx in val_index] 
    y_val_cv = [y_train[idx] for idx in val_index]
    
    X_train_aug, y_train_aug = find_augFiles(X_train_cv,y_train_cv, augImg_folder, augMask_folder)
    X_val_aug,  y_val_aug  = find_augFiles(X_val_cv, y_val_cv, augImg_folder, augMask_folder)

    train_datagen = CustomImageGeneratorTrain(X_train_aug, y_train_aug, patch_xy, b_count)
    val_datagen = CustomImageGeneratorTrain(X_val_aug, y_val_aug, patch_xy, b_count)

    # sanity check
    # batch_nr = random.randint(0, len(train_datagen))
    # X,y = train_datagen[batch_nr]

    # for i in range(X.shape[0]):
        
    #     plt.figure(figsize=(12,6))
    #     plt.subplot(121)
    #     plt.imshow(X[i][:,:,:3])
    #     plt.subplot(122)
    #     plt.imshow(y[i])
    #     plt.show()
    #     plt.savefig(os.path.join(output_folder, "crops/sanity_check{}.png".format(i)) 

    #Load model
    model = binary_unet(patch_xy[0], patch_xy[1], b_count)  

    # Model fit 
    #log_dir = os.path.join(model_dir, "logs") 
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint_path = os.path.join(model_dir, "checkpoints", f"best_weights_k{fold_var}.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_recall", mode='max', verbose=1, save_best_only=True, save_weights_only=True)

    # metrics 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss_function, metrics=["accuracy",
                        tf.keras.metrics.Recall(name="recall"),
                        tf.keras.metrics.Precision(name="precision"),
                        tf.keras.metrics.BinaryIoU(name="iou")])

    model.fit(train_datagen, validation_data=val_datagen, verbose=1, epochs=epochs, callbacks=[checkpoint_callback])

    model.load_weights(checkpoint_path)
    
    print("Evaluate on validation + test data ")
    val_results = model.evaluate(val_datagen)
    test_results = model.evaluate(test_datagen)
    val_r_dict = dict(zip(model.metrics_names,val_results))
    test_r_dict = dict(zip(model.metrics_names, test_results))

    val_metrics[f"Kfold_{fold_var}"] = val_r_dict 
    test_metrics[f"Kfold_{fold_var}"] = test_r_dict

    tf.keras.backend.clear_session()

    fold_var += 1

file = os.path.join(model_dir, "metrics.txt")
if os.path.exists(file):
    open(file, "w").close() # empty textfile for new input
else:
    with open(file, "w") as document: pass # create empty textfile

createMetrics(file, val_metrics, "Validation data")
createMetrics(file, test_metrics, "Test data")

# Find best model comparing one metric from all kfolds and save it 
list_metrics = []

for kfold, metric_dict in test_metrics.items():
    for metric_name, metric_value in metric_dict:
        print(metric_name, "->", metric_value)
        if metric_name == "recall":
            list_metrics.append(metric_value)

max_value = max(list_metrics)
max_index = list_metrics.index(max_value)
             
checkpoint_path = os.path.join(model_dir, "checkpoints", f"best_weights_k{max_index}.h5")

model.load_weights(checkpoint_path)
model.save(os.path.join(model_dir, "best_model"))

# Evaluate the best model again on held out test dataset
print("Test model with best weights:")
model.evaluate(test_datagen)

# Predict the test data with the best model 
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