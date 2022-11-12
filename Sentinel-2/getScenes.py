import os
from glob import glob
import yaml
import numpy as np
from tools_preprocess import ScenceFinderAOI, filterSen12, filterSen2
from matplotlib import pyplot as plt
import shutil

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)

        # preprocess for training
        aoi_path = data['shapefiles']['AOI']
        solar_path = data['shapefiles']['Solar']
        sentinel2_name = data['satellite']['Sentinel2']['name']
        sentinel2_pl = data['satellite']['Sentinel2']['processing_level']
        sentinel2_pt = data['satellite']['Sentinel2']['product_type']
        start_date = data['satellite']['search_criteria']['start_date']
        end_date = data['satellite']['search_criteria']['end_date']
        cloud_cover = data['satellite']['search_criteria']['cloud_cover']
    
        output_folder = data["output_folder"]

# get path of sentinel 1 and 2 tiles on codede server
sen2_scenes = ScenceFinderAOI(aoi_path, sentinel2_name, sentinel2_pl, sentinel2_pt, start_date, end_date, cloud_cover)

Sen2_tiles = filterSen2(sen2_scenes, filterDate=False)
Sen2_tiles.sort()

print("Found the following Sentinel 2 scenes")
[print(tile) for tile in Sen2_tiles]

file = os.path.join(output_folder, "sentinel2_tiles.txt")

if os.path.exists(file):
  os.remove(file)

with open(file, "w") as f:
    for tile in Sen2_tiles:
        f.write("{}\n".format(tile))
