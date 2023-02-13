import os

dir_train_img = "/home/hoehn/data/output/Sentinel-2/crops128/idx/train/img"
dir_train_mask = "/home/hoehn/data/output/Sentinel-12/crops128/idx/test/img"
img_count = len(os.listdir(dir_train_img))
mask_count = len(os.listdir(dir_train_mask))

print("Is the length of image folder same as for the mask folder?")
print("{}={}: {}".format(img_count, mask_count,(img_count==mask_count)))

equal = []
missing = []

for i in os.listdir(dir_train_img):
    #i = "_".join(["mask" if j == "img" else j for j in i.split("_")]) 

    if i not in os.listdir(dir_train_mask):
        missing.append(i)
    else:
        equal.append(i)

print("Found {} equal files in both dirs".format(len(equal)))
print("Found {} missing files in both dirs".format(len(missing)))
