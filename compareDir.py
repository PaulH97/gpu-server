import os 

dir1 = "/home/hoehn/data/output/Sentinel-12/crops/idx/train/img"
dir2 = "/home/hoehn/data/output/Sentinel-12/crops/idx/train/mask"

print(len(os.listdir(dir1)), len(os.listdir(dir2)))

for file in os.listdir(dir1):
    if "aug" in file:
        os.remove(os.path.join(dir1, file))

for file in os.listdir(dir2):
    if "aug" in file:
        os.remove(os.path.join(dir2, file))

print(len(os.listdir(dir1)), len(os.listdir(dir2)))


# equal = []
# missing = []

# for i in dir1:
#     i = "_".join(["mask" if i == "img" else i for i in i.split("_")]) 
#     print(i)
#     if i not in dir2:
#         missing.append(i)
#     else:
#         equal.append(i)

# print("Found {} equal files in both dirs".format(len(equal)))
# print("Found {} missing files in both dirs".format(len(missing)))
# print(len(dir1))
# print(len(dir2))
