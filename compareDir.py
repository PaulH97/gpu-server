import os 

dir1 = os.listdir("/home/hoehn/data/output/Sentinel-1/crops/idx/mask")
dir2 = os.listdir("/home/hoehn/data/output/Sentinel-1/crops/idx/img")

equal = []
missing = []

for i in dir1:
    if i not in dir2:
        missing.append(i)
    else:
        equal.append(i)

print("Found {} equal files in both dirs".format(len(equal)))
print("Found {} missing files in both dirs".format(len(missing)))
print(len(dir1))
print(len(dir2))
