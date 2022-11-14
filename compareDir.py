import os 

dir1 = os.listdir("/home/hoehn/data/output/Sentinel-2/crops/idx/mask")
dir2 = os.listdir("/home/hoehn/data/output/Sentinel-12/crops/idx/mask")

equal = []
missing = []

for i in dir1:
    if i not in dir2:
        missing.append(i)
    else:
        equal.append(i)

print("Found {} equal files in both dirs".format(len(equal)))
print("Found {} missing files in both dirs".format(len(missing)))
