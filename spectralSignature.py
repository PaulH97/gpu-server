import os
import yaml
from rasterstats import zonal_stats
from tools_preprocess import getBandPaths, ScenceFinderAOI
import rasterio
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Read data from config file
if os.path.exists("config.yaml"):
    with open('config.yaml') as f:

        data = yaml.load(f, Loader=yaml.FullLoader)
        
        aoi_path = data['shapefiles']['AOI_specSig']
        landuse_path = data['shapefiles']['landuse']
        
        sentinel1_name = data['satellite']['Sentinel1']['name']
        sentinel1_pt = data['satellite']['Sentinel1']['product_type']
        sentinel1_pl = data['satellite']['Sentinel1']['processing_level']
        
        sentinel2_name = data['satellite']['Sentinel2']['name']
        sentinel2_pl = data['satellite']['Sentinel2']['processing_level']
        sentinel2_pt = data['satellite']['Sentinel2']['product_type']
        
        start_date = data['satellite']['search_criteria']['start_date']
        end_date = data['satellite']['search_criteria']['end_date']
        cloud_cover = data['satellite']['search_criteria']['cloud_cover']
    
        output_folder = data["output_folder"]

scenes = ScenceFinderAOI(aoi_path, sentinel2_name, sentinel2_pl, sentinel2_pt, start_date, end_date, cloud_cover)
scenes += ["/codede/Sentinel-2/MSI/L3-WASP/2021/07/01/S2_L3_WASP_202107_33UUU"]

vector = gpd.read_file(landuse_path)
df_base = pd.DataFrame(columns=["B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "month"])

for tile_month in scenes:

    month = "-".join(tile_month.split(os.sep)[5:7])
    print("Extracting values of month: ", month)

    bands = getBandPaths(tile_month, "Sentinel-2")

    # dict for pandas dataframe col=Bands row=band mean value per landuse
    bands_avg = {"B11": [], "B12": [], "B2": [], "B3": [], "B4": [], "B5": [], "B6": [], "B7": [], "B8": [], "B8A": []}
    
    for band in bands:

        band_name = os.path.basename(band).split(".")[0].split("_")[-1]
        print("Start with band: ", band_name)
        
        raster = rasterio.open(band)
        raster_crs = raster.crs        
        vector_reproj = vector.to_crs(epsg=raster_crs.to_epsg()) 

        stats = zonal_stats(vector_reproj, band, geojson_out=True)
        # iterate over features of landuse shapefile
        data_lu = {"solar": [], "grassland": [], "forest": [], "sealedSurfaces": [], "agriculture": []}

        for feature in stats:

            if feature["properties"]["landuse"]== "solar":
                data_lu["solar"].append(feature["properties"]["mean"])

            elif feature["properties"]["landuse"]== "grassland":
                data_lu["grassland"].append(feature["properties"]["mean"])
            
            elif feature["properties"]["landuse"]== "forest":
                data_lu["forest"].append(feature["properties"]["mean"])
            
            elif feature["properties"]["landuse"]== "sealedSurfaces":
                data_lu["sealedSurfaces"].append(feature["properties"]["mean"])
            else:
                data_lu["agriculture"].append(feature["properties"]["mean"])
        
        data_avg = []
        for key, value in data_lu.items():

            data_avg.append((sum(filter(None, value))/len(value)))

        bands_avg[band_name] = data_avg
    
    bands_avg["month"] = [month for i in range(len(bands_avg["B11"]))]
    #df_tile = pd.DataFrame(data=bands_avg)
    df_base = df_base.append(pd.DataFrame(data=bands_avg))

table_path = os.path.join(output_folder, "specSignature.csv")
df_base.to_csv(table_path)

