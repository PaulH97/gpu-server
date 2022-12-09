import os
from glob import glob
import yaml
from tools_model import dice_coef, load_img_as_array
from sklearn.model_selection import train_test_split
from unet import binary_unet
from matplotlib import pyplot as plt
from datagen import CustomImageGenerator
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import random

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
    img_list = glob("{}/crops/idx/img/*.tif".format(output_folder))
    mask_list = glob("{}/crops/idx/mask/*.tif".format(output_folder))
    model_name = parameter[optimizer] + "_" + parameter[loss_function] + "_" + str(epochs) + "_idx" # Ad_bc_100_idx
else:
    img_list = glob("{}/crops/no_idx/img/*.tif".format(output_folder))
    mask_list = glob("{}/crops/no_idx/mask/*.tif".format(output_folder))
    model_name = parameter[optimizer] + "_" + parameter[loss_function] + "_" + str(epochs) + "no_idx" # Ad_bc_100

model_path = os.path.join(output_folder, "models", model_name)

img_list.sort()
mask_list.sort()

# Split training data
X_train, X_test, y_train, y_test = train_test_split(img_list, mask_list, test_size = 0.20, shuffle=True, random_state = seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.10, shuffle=True, random_state = seed)

# check if spliting data
countX = 0
county = 0

for img in X_val:
    if img in X_train:
        countX += 1

for img in y_val:
    if img in y_train:
        county += 1

print("Found {} validation images in train data".format(countX))
print("Found {} validation masks in train data".format(county))

# Load images and masks with an custom data generator - for performance reason
patch_array = load_img_as_array(X_train[0])
patch_xy = (patch_array.shape[0], patch_array.shape[1])
b_count = patch_array.shape[-1]

train_datagen = CustomImageGenerator(X_train, y_train, patch_xy, b_count)
val_datagen = CustomImageGenerator(X_val,y_val, patch_xy, b_count)
test_datagen = CustomImageGenerator(X_test, y_test, patch_xy, b_count)

#sanity check
batch_nr = random.randint(0, len(train_datagen))
X,y = train_datagen[batch_nr]

for i in range(X.shape[0]):

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(X[i][:,:,])
    plt.subplot(122)
    plt.imshow(y[i])
    plt.show()
    plt.savefig("sanity_check{}.png".format(i)) 
                                                                                                                                                                                                                                                                                                                                                        
#Load model
model = binary_unet(patch_xy[0], patch_xy[1], b_count)  

# metrics 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=loss_function, metrics=[dice_coef,
                    tf.keras.metrics.BinaryAccuracy(name ="accuracy"),
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.BinaryIoU(name="iou")])

# Model fit 
log_dir = os.path.join(output_folder, "models", "logs", model_name) 
checkpoint_path = os.path.join(output_folder, "models", "checkpoints", model_name)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor="val_iou", mode='max', verbose=1, save_best_only=True, save_weights_only=True)
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_iou", patience=5, verbose=1)

model.fit(train_datagen, validation_data=val_datagen, verbose=1, epochs=epochs, callbacks=[checkpoint_callback, earlyStopping])

# Save model for prediction
model.save(model_path)

model.load_weights(checkpoint_path)
model.evaluate(test_datagen)

pred_test = model.predict(test_datagen) # f.eg.(288,128,128,1)
pred_test = (pred_test > 0.5).astype(np.uint8) 

for i in range((test_datagen[0][1].shape[0])):

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(test_datagen[0][1][i])
    plt.subplot(122)
    plt.imshow(pred_test[i])
    plt.show()
    plt.savefig("/home/hoehn/data/prediction/prediction{}.png".format(i))

# k fold cross validation

# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((target_train, target_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  # Define the model architecture
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))

  # Compile the model
  model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1
