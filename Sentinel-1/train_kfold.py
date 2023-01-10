import os
from glob import glob
import yaml
from tools_model import dice_coef, load_img_as_array, createMetrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from unet import binary_unet
from matplotlib import pyplot as plt
from datagen import CustomImageGenerator
import tensorflow as tf
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
        k_folds = data['model_parameter']['kfold']

        seed = data["seed"]

# Use patches as trainings data for model
parameter = {"Adam": "Ad", "binary_crossentropy": "bc"}

if indizes: 
    img_list = glob("{}/crops/idx/img/*.tif".format(output_folder))
    mask_list = glob("{}/crops/idx/mask/*.tif".format(output_folder))
    model_name = parameter[optimizer] + "_" + parameter[loss_function] + "_" + str(epochs) + "_idx" # Ad_bc_100_idx
else:
    img_list = glob("{}/crops/no_idx/img/*.tif".format(output_folder))
    mask_list = glob("{}/crops/no_idx/mask/*.tif".format(output_folder))
    model_name = parameter[optimizer] + "_" + parameter[loss_function] + "_" + str(epochs) + "no_idx" # Ad_bc_100

model_dir = os.path.join(output_folder, "models", model_name)

if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)

os.mkdir(model_dir)
os.mkdir(os.path.join(model_dir, "checkpoints"))
os.mkdir(os.path.join(model_dir, "logs"))

img_list.sort()
mask_list.sort()

# Split training data
# X_train, X_test, y_train, y_test = train_test_split(img_list, mask_list, test_size = 0.20, shuffle=True, random_state = seed)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.10, shuffle=True, random_state = seed)

# Split training data in training-test set 
X_train, X_test, y_train, y_test = train_test_split(img_list, mask_list, test_size = 0.20, shuffle=True, random_state = seed)
print(f"Training data: {len(X_train)} Test dataset: {len(X_test)}")

# Load images and masks with an custom data generator - for performance reason
patch_array = load_img_as_array(X_train[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

test_datagen = CustomImageGenerator(X_test, y_test, patch_xy, b_count)

## Training with K-fold cross validation
kf = KFold(n_splits=k_folds, random_state=seed, shuffle=True)
fold_var = 1

val_metrics = {}
test_metrics = {}

# Model fit 
#log_dir = os.path.join(model_dir, "logs") 
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
checkpoint_path = os.path.join(model_dir, "checkpoints", "best_weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_iou", mode='max', verbose=1, save_best_only=True, save_weights_only=True)

for i, (train_index, val_index) in enumerate(kf.split(X_train)):

    # List of paths to images based on train_index
    X_train_cv = [X_train[idx] for idx in train_index]
    y_train_cv = [y_train[idx] for idx in train_index]
    X_val_cv = [X_train[idx] for idx in val_index]
    y_val_cv = [y_train[idx] for idx in val_index]
    
    train_datagen = CustomImageGenerator(X_train_cv, y_train_cv, patch_xy, b_count)
    val_datagen = CustomImageGenerator(X_val_cv, y_val_cv, patch_xy, b_count)

    # sanity check
    # X,y = train_datagen[0]

    # for i in range(X.shape[0]):

    #     plt.figure(figsize=(12,6))
    #     plt.subplot(121)
    #     plt.imshow(X[i][:,:,2:5])
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

    model.fit(train_datagen, validation_data=val_datagen, verbose=1, epochs=epochs, callbacks=[checkpoint_callback])

    print("Evaluate on validation + test data ")
    val_results = model.evaluate(val_datagen)
    test_results = model.evaluate(test_datagen)
    val_r_dict = dict(zip(model.metrics_names,val_results))
    test_r_dict = dict(zip(model.metrics_names, test_results))

    val_metrics[f"Kfold_{fold_var}"] = val_r_dict 
    test_metrics[f"Kfold_{fold_var}"] = test_r_dict

    tf.keras.backend.clear_session()

    fold_var += 1

# Save best model
model.load_weights(checkpoint_path)
model.save(os.path.join(model_dir, "best_model"))

print("Test model with best weights:")
model.evaluate(test_datagen)

file = os.path.join(model_dir, "metrics.txt")
if os.path.exists(file):
    open(file, "w").close() # empty textfile for new input
else:
    with open(file, "w") as document: pass # create empty textfile

createMetrics(file, val_metrics, "Validation data")
createMetrics(file, test_metrics, "Test data")

# pred_test = model.predict(test_datagen) # f.eg.(288,128,128,1)
# pred_test = (pred_test > 0.5).astype(np.uint8) 

# for i in range((test_datagen[0][2].shape[0])):

#     plt.figure(figsize=(12,6))
#     plt.subplot(121)
#     plt.imshow(test_datagen[0][1][i])
#     plt.subplot(122)
#     plt.imshow(pred_test[i])
#     plt.savefig("/home/hoehn/data/prediction/prediction{}.png".format(i))
