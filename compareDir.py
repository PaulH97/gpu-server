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

count = 0 
missing2 = []

for i in missing:
    if "nopv" in i:
        count += 1
    else:
        missing2.append(i)

print(len(equal))
print(len(missing))
[print(i) for i in missing2]

import pdb
pdb.set_trace()
