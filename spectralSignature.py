import os
import yaml
from rasterstats import zonal_stats
from tools_preprocess import getBandPaths, ScenceFinderAOI
import rasterio
import geopandas as gpd

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
vector_crs = vector.crs

for tile_month in scenes:

    month = "-".join(tile_month.split(os.sep)[5:7])
    print("Extracting values of month: ", month)

    bands = getBandPaths(tile_month, "Sentinel-2")
    for band in bands:
        
        raster = rasterio.open(band)
        raster_crs = raster.crs        
        vector_32633 = vector.to_crs(epsg=raster_crs.to_epsg()) 

        stats = zonal_stats(landuse_path, band, geojson_out=True)
        # iterate over features of landuse shapefile
    
        import pdb 
        pdb.set_trace()






