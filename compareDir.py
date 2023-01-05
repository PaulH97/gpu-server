import os 

dir1 = os.listdir("/home/hoehn/data/output/Sentinel-2/crops/idx/train/img")
dir2 = os.listdir("/home/hoehn/data/output/Sentinel-2/crops/idx/train/mask")

equal = []
missing = []

for i in dir1:
    i = "_".join(["mask" if i == "img" else i for i in i.split("_")]) 
    print(i)
    if i not in dir2:
        missing.append(i)
    else:
        equal.append(i)

print("Found {} equal files in both dirs".format(len(equal)))
print("Found {} missing files in both dirs".format(len(missing)))
print(len(dir1))
print(len(dir2))
