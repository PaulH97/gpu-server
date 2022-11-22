import os
from glob import glob
import yaml
import numpy as np
from tools_preprocess import ScenceFinderAOI, filterSen1
from matplotlib import pyplot as plt
import shutil

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)

        # preprocess for training
        aoi_path = data['shapefiles']['AOI']
        solar_path = data['shapefiles']['Solar']
        sentinel1_name = data['satellite']['Sentinel1']['name']
        sentinel1_pl = data['satellite']['Sentinel1']['processing_level']
        sentinel1_pt = data['satellite']['Sentinel1']['product_type']
        start_date = data['satellite']['search_criteria']['start_date']
        end_date = data['satellite']['search_criteria']['end_date']
        cloud_cover = data['satellite']['search_criteria']['cloud_cover']
    
        output_folder = data["output_folder"]

# get path of sentinel 1 and 2 tiles on codede server
sen2_scenes = ScenceFinderAOI(aoi_path, sentinel1_name, sentinel1_pl, sentinel1_pt, start_date, end_date, cloud_cover)

Sen2_tiles = filterSen1(sen2_scenes, filterDate=False)
Sen2_tiles.sort()

print("Found the following Sentinel 1 scenes")
[print(tile) for tile in Sen2_tiles]

file = os.path.join(output_folder, "sentinel1_tiles.txt")

if os.path.exists(file):
  os.remove(file)

with open(file, "w") as f:
    for tile in Sen2_tiles:
        f.write("{}\n".format(tile))
