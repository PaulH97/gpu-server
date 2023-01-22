import yaml
import os
from tools_model import load_trainData, load_img_as_array, dice_metric
from matplotlib import pyplot as plt
from datagen import CustomImageGeneratorTest
import tensorflow as tf
import numpy as np
import random 
from focal_loss import BinaryFocalLoss
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        indizes = data["indizes"]
        output_folder = data["output_folder"]
        prediction_folder = os.path.join(output_folder, "prediction")
        best_model = data["evaluation"]["best_model"]

evaluate_test = False
evaluate_pred = True

# Save predicted images from one batch in batch folder 
batch_folder = os.path.join(prediction_folder, "batch_prediction")

if not os.path.isdir(batch_folder):
    os.mkdir(batch_folder)

if evaluate_test: 
    # Load training + test data from local folder and sort paths after preprocessing 
    X_train, X_test, y_train, y_test = load_trainData(output_folder, indizes)

    print(f"Training dataset: {len(X_train)} Test dataset: {len(X_test)}")

    # Load images and masks with an custom data generator - for performance reason
    patch_array = load_img_as_array(X_test[0])
    patch_xy = (patch_array.shape[0], patch_array.shape[1])
    b_count = patch_array.shape[-1]

    # Completly held out data as generator - for later evaluation
    test_datagen = CustomImageGeneratorTest(X_test, y_test, patch_xy, b_count)

    # Load best model from path
    model = tf.keras.models.load_model(best_model, compile=False, custom_objects=[dice_metric])

    # Predict the test data with the best model 
    pred_test = model.predict(test_datagen) # f.eg.(288,128,128,1)
    pred_test = (pred_test > 0.5).astype(np.uint8) 

    batch_size = test_datagen[0][1].shape[0]
    batch_nr = random.randint(0, len(test_datagen)-1)

    # test_datagen[0] = ((16,128,128,12),(16,128,128,1)) -> tupel of length 2
    for i in range(batch_size): 
        
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(test_datagen[batch_nr][1][i]) 
        plt.subplot(122)
        j = i+(batch_nr * batch_size)
        plt.imshow(pred_test[j])
        plt.savefig("{}/batchNr{}_pred{}.png".format(batch_folder, batch_nr, i)) 

if evaluate_pred:
    # ----------------- Train PV crops of prediction tile (not full img) ---------------------------
    X_pred = glob("{}/*.tif".format(os.path.join(prediction_folder, "crops", "img")))
    y_pred = glob("{}/*.tif".format(os.path.join(prediction_folder, "crops", "mask")))

    X_pred.sort()
    y_pred.sort()

    # Load images and masks with an custom data generator - for performance reason
    patch_array = load_img_as_array(X_pred[0])
    patch_xy = (patch_array.shape[0], patch_array.shape[1])
    b_count = patch_array.shape[-1]

    # Completly held out data as generator - for later evaluation
    test_datagen = CustomImageGeneratorTest(X_pred, y_pred, patch_xy, b_count)

    # Load best model from path
    model = tf.keras.models.load_model(best_model, compile=False)

    # Predict the test data with the best model 
    pred_test = model.predict(test_datagen) # f.eg.(32,128,128,1)
    pred_test = (pred_test > 0.5).astype(np.uint8) 

    batch_size = test_datagen[0][1].shape[0]
    batch_nr = random.randint(0, len(test_datagen)-1)

    # test_datagen[0] = ((16,128,128,12),(16,128,128,1)) -> tupel of length 2
    for i in range(batch_size): 
        
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(test_datagen[batch_nr][1][i]) 
        plt.subplot(122)
        j = i+(batch_nr * batch_size)
        plt.imshow(pred_test[j])
        plt.savefig("{}/batchNr{}_predPV{}.png".format(batch_folder, batch_nr, i))
        