import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from tools_preprocess import load_img_as_array

# train = pd.read_csv("/home/hoehn/data/output/Sentinel-2/logs_train_recall.csv")
# val = pd.read_csv("/home/hoehn/data/output/Sentinel-2/logs_validation_recall.csv")

# plt.plot(train["Step"], train["Value"], label="Train", color="royalblue")
# plt.plot(val["Step"], val["Value"],  label="Validation", color="darkorange")
# plt.title("Learning curves of model metric: recall", fontsize = 14)
# plt.ylabel('Recall', fontsize = 10)
# plt.xlabel('Epochs', fontsize = 10)
# plt.legend()
# plt.show()
# plt.savefig("recall.png")

X = load_img_as_array("/home/hoehn/data/output/Sentinel-2/crops128/idx/train/img/33UVT_img_5971_pv.tif")
y = load_img_as_array("/home/hoehn/data/output/Sentinel-2/crops128/idx/train/mask/33UVT_mask_5971_pv.tif")

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.title("False-colour composite (BGR) of patch", fontsize = 12)
plt.imshow(X[:,:,2:5]) # 0:B11 1:B12 2:B2 3:B3 4:B4 ... # VH VV 
plt.subplot(122)
plt.title("Binary Mask of patch", fontsize = 12)
plt.imshow(y)
plt.show()
plt.savefig("sen2.png")
