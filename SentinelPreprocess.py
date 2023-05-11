import os
from glob import glob
import yaml
import numpy as np
from tools_preprocess import *
import rasterio
from matplotlib import pyplot as plt
import shutil
import warnings
from sklearn.model_selection import train_test_split
import ast
import json

# evtl band class
# alle band functionen kommen in eigene FUnktionen von eine Klasse names Band

class SatelliteBand:
    def __init__(self, band_name, band_path):
        self.name = band_name 
        self.path = band_path
        
    def read_band(self):
                            
        band_array = rasterio.open(self.path).read()           
        band_array = np.moveaxis(band_array, 0, -1)
        band_array = np.nan_to_num(band_array)
            
        return band_array                               
    
    def resample_band(self, xres, yres, out_dir):
        
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform, reproject

        # Input und Output TIFF-Dateipfade
        input_tif = self.path
        out_name = f"{self.name}_resampled_{xres}x{yres}"
        output_tif = os.path.join(out_dir, f"{out_name}.tif")

        # Zieldimensionen in Metern
        target_resolution = (xres, yres)

        with rasterio.open(input_tif) as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                src.crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=target_resolution
            )

            meta = src.meta.copy()
            meta.update({
                'crs': src.crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(output_tif, 'w', **meta) as dst:
                for band in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear
                    )
        
        band_resampled = SatelliteBand(out_name, output_tif)
        
        print("Resampled band and saved as follwed: ", output_tif)

        return band_resampled
    
    def plot_band(self):
        
        band_array = self.read_band()

        # band_array = self.read_band(band_id)
        q25, q75 = np.percentile(band_array, [25, 75])
        bin_width = 2 * (q75 - q25) * len(band_array) ** (-1/3)
        bins = round((band_array.max() - band_array.min()) / bin_width)   
        bins = 150
        
        plt.figure(figsize=(10,10))
        plt.title("Band {} - Histogram".format(self.name), fontsize = 20)
        plt.hist(band_array.flatten(), bins = bins, color="lightcoral")
        plt.ylabel('Number of Pixels', fontsize = 16)
        plt.xlabel('DN', fontsize = 16)
        plt.savefig(f"{self.name}_histo.png")
        print("Saved figure in current working folder")

        return
    
    def normalize_band(self, output_dir, norm_textfile):
        
        bands_scale = json.load(open(norm_textfile))
               
        with rasterio.open(self.path) as src:
            band_array = src.read(1)
            meta = src.meta.copy()
                   
            a, b = 0, 1
            c, d = bands_scale[self.name]
            band_array_norm = (b - a) * ((band_array - c) / (d - c)) + a
            band_array_norm[band_array_norm > 1] = 1
            band_array_norm[band_array_norm < 0] = 0
            
            meta.update(dtype=band_array_norm.dtype)
            
            output_path = os.path.join(output_dir, f'{self.name}_normalized.tif')
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(band_array_norm, 1)
        
        band_norm = SatelliteBand(f"{self.name}_norm", output_path)
    
        print("Successfully normalized band ", self.name)

        return band_norm
    
    def create_patches(self, output_dir, patch_size=128):
        
        output_dir = os.path.join(output_dir, "patches")
        output_dirBand = os.path.join(output_dir, f"{self.name}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dirBand, exist_ok=True)
        
        with rasterio.open(self.path) as src:
            img_array = src.read(1)
            meta = src.meta.copy()
            height, width = img_array.shape

            for y in range(0, height, patch_size):
                for x in range(0, width, patch_size):
                    patch = img_array[y:y+patch_size, x:x+patch_size]

                    if patch.shape == (patch_size, patch_size):
                        # Update metadata
                        meta.update(width=patch_size, height=patch_size)

                        # Save patches
                        output_path = os.path.join(output_dirBand, f'{y}_{x}.tif')
                        with rasterio.open(output_path, 'w', **meta) as dst:
                            dst.write(patch, 1)
        
        print("Saved patches in folder: ", output_dirBand)
                            
        return output_dir
    
class SatelliteImage:
    def __init__(self, tile_id, month, year, bands=None):
        self.tile_id = tile_id # 32UMD
        self.month = month 
        self.year = year
        self.bands = bands or {}
        self.seed = 42
    
    def date_to_string(self, delimiter=''):
        return f"{self.year:04d}{delimiter}{self.month:02d}"
    
    def add_band(self, band_id, file_path):
        self.bands[band_id] = SatelliteBand(band_id, file_path)
        
    def get_band(self, band_id):
        return self.bands.get(band_id)

    def __getattr__(self, attr_name):
        # Versuchen, das Band als Attribut abzurufen
        band = self.get_band(attr_name.upper())
        if band:
            return band
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")  
              
    def initiate_bands(self):
        
        import re

        def extract_band_info(file_path):
            # Regular expression to match the band information
            pattern = r"_B(\d+[A-Z]?)\.tif"

            # Search for the pattern in the file path
            match = re.search(pattern, file_path)

            if match:
                band_info = "B" + match.group(1)
                return band_info
            else:
                return None
        
        bands_path = glob(f"{self.tile_folder}/*.tif") 
        bands_path.sort()  # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A    

        # Extract band information from the list of file paths using a loop
        for band_path in bands_path:
            band_id = extract_band_info(band_path)
            if band_id:       
                self.add_band(band_id, band_path)
    
        print(f"The corresponding bands of the {self.satellite} misson have been added")
        
        return
          
    def sanity_check(self, patch_folder):
        
        def get_sorted_images(directory):
            # List the content of the directory and filter only files
            content = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            # Sort the content
            content.sort()
            
            return content

        def pick_random_image(images):
            random.seed(self.seed)
            return random.choice(images)        
        
        patchDir_content = os.listdir(patch_folder)
        # look for folder train and mask 
        if "img" in patchDir_content and "mask" in patchDir_content: 
            
            # Replace 'your_img_directory_path' and 'your_mask_directory_path' with the actual paths
            img_directory = os.path.isfile(os.path.join(patch_folder, "img"))
            mask_directory = os.path.isfile(os.path.join(patch_folder, "mask"))

            img_files = get_sorted_images(img_directory)
            mask_files = get_sorted_images(mask_directory)

            if img_files and mask_files:
                random_img_path = pick_random_image(img_files)
                random_mask_path = pick_random_image(mask_files)
                
                random_img_name = os.path.basename(random_img_path).split(".")[0]
                
                random_img = rasterio.open(random_img_path)
                random_mask = rasterio.open(random_mask_path)
                
                image_path = os.path.join(patch_folder, f"sanityCheck-{random_img_name}.png")
                plt.figure(figsize=(12,6))
                plt.subplot(121)
                plt.imshow(random_img) # image 
                plt.subplot(122)
                plt.imshow(random_mask) # mask 
                plt.savefig(image_path) 

            else:
                print("One or both directories are empty.")
        else: 
            print("Img and mask directories are missing in patch folder.")
                
        return 
    
    def patches_in_raster(self, patches, output_folder):
        # input needs to be an dictonary of bands as key and value as patches array
        # can be dict of single mask band or multiple bands 
    
        band_names = list(patches.keys())
        band_names.sort()

        for idx in range(len(patches[band_names[0]])):

            final = rasterio.open(os.path.join(output_folder, f'img_{idx}.tif'),'w', driver='Gtiff',
                            width=patches[band_names[0]][0].shape[0], height=patches[band_names[0]][0].shape[1],
                            count=len(band_names),
                            dtype=rasterio.float64)

            for band_nr, band_name in enumerate(band_names):
                final.write(patches[band_name][idx][:,:,0],band_nr+1)
                final.set_band_description(band_nr+1, band_name)
            final.close()

        return 

    def preprocess_tile(self, norm_textfile, out_dir, training=True): 
                
        if not self.indizes_path:
            print("Preprocessing without indizes")
        else: 
            print("Preprocessing with indizes")
                   
        for band in self.bands.values(): # band = SatelliteBand()
            
            print("Start with band: ", band.name)
            band_norm = band.normalize_band(out_dir, norm_textfile)
            band_norm.create_patches(out_dir)
            
            import pdb
            pdb.set_trace()

# class SatelliteImage:
#     def __init__(self, tile_id, month, year):
#         self.tile_id = tile_id # 32UMD
#         self.month = month 
#         self.year = year
#         self.seed = 42
    
#     def date_to_string(self, delimiter=''):
#         return f"{self.year:04d}{delimiter}{self.month:02d}"
    
#     def get_band_path(self, band_id):
#         bands_path = self.get_bands_path()
#         band_path = [i for i in bands_path if i.find(band_id) > 0][-1]
#         return band_path
        
#     def get_bands_path(self):
#         # Hole den Pfad und den Namen für jede Band        
#         bands_path = glob(f"{self.tile_folder}/*.tif") + self.indizes_path
#         bands_path.sort()  # B11 B12 B2 B3 B4 B5 B6 B7 B8 B8A
#         return bands_path
   
#     def create_patches(self, band_id, output_dir, patch_size=128):
        
#         out_dir = os.path.join(output_dir, "patches")
#         os.makedirs(out_dir, exist_ok=True)
        
#         band_path = self.get_band_path(band_id)
           
#         with rasterio.open(band_path) as src:
#             img_array = src.read(1)
#             meta = src.meta.copy()
#             height, width = img_array.shape

#             for y in range(0, height, patch_size):
#                 for x in range(0, width, patch_size):
#                     patch = img_array[y:y+patch_size, x:x+patch_size]

#                     if patch.shape == (patch_size, patch_size):
#                         # Update metadata
#                         meta.update(width=patch_size, height=patch_size)

#                         # Save patch
#                         output_path = os.path.join(output_dir, f'{band_id}_{y}_{x}.tif')
#                         with rasterio.open(output_path, 'w', **meta) as dst:
#                             dst.write(patch, 1)
        
#         print("Saved patches in folder: ", out_dir)
                            
#         return
        
#     def patchify_patches(self, array, patch_size):

#         patches = patchify(array, (patch_size, patch_size, 1), step=patch_size)    
#         patchX = patches.shape[0]
#         patchY = patches.shape[1]
#         result = []
#         #scaler = MinMaxScaler()

#         for i in range(patchX):
#             for j in range(patchY):
#                 single_patch_img = patches[i,j,:,:]
#                 #single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
#                 single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
#                 result.append(single_patch_img)
        
#         return result   
       
#     def resample_band(self, band_id, xres, yres, out_dir):
        
#         import rasterio
#         from rasterio.enums import Resampling
#         from rasterio.warp import calculate_default_transform, reproject

#         # Input und Output TIFF-Dateipfade
#         input_tif = self.get_band_path(band_id)
#         output_tif = os.path.join(out_dir, f"{band_id}_resampled_{xres}x{yres}.tif")

#         # Zieldimensionen in Metern
#         target_resolution = (xres, yres)

#         with rasterio.open(input_tif) as src:
#             transform, width, height = calculate_default_transform(
#                 src.crs,
#                 src.crs,
#                 src.width,
#                 src.height,
#                 *src.bounds,
#                 resolution=target_resolution
#             )

#             meta = src.meta.copy()
#             meta.update({
#                 'crs': src.crs,
#                 'transform': transform,
#                 'width': width,
#                 'height': height
#             })

#             with rasterio.open(output_tif, 'w', **meta) as dst:
#                 for band in range(1, src.count + 1):
#                     reproject(
#                         source=rasterio.band(src, band),
#                         destination=rasterio.band(dst, band),
#                         src_transform=src.transform,
#                         src_crs=src.crs,
#                         dst_transform=transform,
#                         dst_crs=src.crs,
#                         resampling=Resampling.bilinear
#                     )
        
#         print("Resampled band and saved as follwed: ", output_tif)
                 
#         return 
    
#     def read_band(self, band_id):
                               
#         band_path = self.get_band_path(band_id)
#         print("Reading band: ", band_id)
        
#         band_array = rasterio.open(band_path).read()           
#         band_array = np.moveaxis(band_array, 0, -1)
#         band_array = np.nan_to_num(band_array)
            
#         return band_array                               
    
#     def plot_band(self, band_array, band_name):

#         # band_array = self.read_band(band_id)
#         q25, q75 = np.percentile(band_array, [25, 75])
#         bin_width = 2 * (q75 - q25) * len(band_array) ** (-1/3)
#         bins = round((band_array.max() - band_array.min()) / bin_width)   
#         bins = 150
        
#         plt.figure(figsize=(10,10))
#         plt.title("Band {} - Histogram".format(band_name), fontsize = 20)
#         plt.hist(band_array.flatten(), bins = bins, color="lightcoral")
#         plt.ylabel('Number of Pixels', fontsize = 16)
#         plt.xlabel('DN', fontsize = 16)
#         plt.savefig(f"{band_name}_histo.png")
#         print("Saved figure in current working folder")

#         return
    
#     def sanity_check(self, patch_folder):
        
#         def get_sorted_images(directory):
#             # List the content of the directory and filter only files
#             content = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
#             # Sort the content
#             content.sort()
            
#             return content

#         def pick_random_image(images):
#             random.seed(self.seed)
#             return random.choice(images)        
        
#         patchDir_content = os.listdir(patch_folder)
#         # look for folder train and mask 
#         if "img" in patchDir_content and "mask" in patchDir_content: 
            
#             # Replace 'your_img_directory_path' and 'your_mask_directory_path' with the actual paths
#             img_directory = os.path.isfile(os.path.join(patch_folder, "img"))
#             mask_directory = os.path.isfile(os.path.join(patch_folder, "mask"))

#             img_files = get_sorted_images(img_directory)
#             mask_files = get_sorted_images(mask_directory)

#             if img_files and mask_files:
#                 random_img_path = pick_random_image(img_files)
#                 random_mask_path = pick_random_image(mask_files)
                
#                 random_img_name = os.path.basename(random_img_path).split(".")[0]
                
#                 random_img = rasterio.open(random_img_path)
#                 random_mask = rasterio.open(random_mask_path)
                
#                 image_path = os.path.join(patch_folder, f"sanityCheck-{random_img_name}.png")
#                 plt.figure(figsize=(12,6))
#                 plt.subplot(121)
#                 plt.imshow(random_img) # image 
#                 plt.subplot(122)
#                 plt.imshow(random_mask) # mask 
#                 plt.savefig(image_path) 

#             else:
#                 print("One or both directories are empty.")
#         else: 
#             print("Img and mask directories are missing in patch folder.")
                
#         return 
    
#     def normalize_band(self, band_id, output_dir, norm_textfile):
        
#         bands_scale = json.load(open(norm_textfile))
#         band_path = self.get_band_path(band_id)
           
#         with rasterio.open(band_path) as src:
#             band_array = src.read(1)
#             meta = src.meta.copy()
            
#             print(f"max{band_array.max()}, min{band_array.min()}, mean{band_array.mean()}")
        
#             a, b = 0, 1
#             c, d = bands_scale[band_id]
#             band_array_norm = (b - a) * ((band_array - c) / (d - c)) + a
#             band_array_norm[band_array_norm > 1] = 1
#             band_array_norm[band_array_norm < 0] = 0
            
#             meta.update(dtype=band_array_norm.dtype)
            
#             output_path = os.path.join(output_dir, f'{band_id}_normalized.tif')
#             with rasterio.open(output_path, 'w', **meta) as dst:
#                 dst.write(band_array_norm, 1)
        
#         print("Successfully normalized band ", band_id)

#         return band_array_norm
    
#     def patches_in_raster(self, patches, output_folder):
#         # input needs to be an dictonary of bands as key and value as patches array
#         # can be dict of single mask band or multiple bands 
    
#         band_names = list(patches.keys())
#         band_names.sort()

#         for idx in range(len(patches[band_names[0]])):

#             final = rasterio.open(os.path.join(output_folder, f'img_{idx}.tif'),'w', driver='Gtiff',
#                             width=patches[band_names[0]][0].shape[0], height=patches[band_names[0]][0].shape[1],
#                             count=len(band_names),
#                             dtype=rasterio.float64)

#             for band_nr, band_name in enumerate(band_names):
#                 final.write(patches[band_name][idx][:,:,0],band_nr+1)
#                 final.set_band_description(band_nr+1, band_name)
#             final.close()

#         return 

#     def preprocess_tile(self, patch_size, norm_textfile, out_dir, training=True): 
        
#         bands_path = self.get_bands_path()
        
#         if not self.indizes_path:
#             print("Preprocessing without indizes")
#         else: 
#             print("Preprocessing with indizes")
                
#         bands_patches = {}
    
#         # read_band and resample band muss noch gecheckt werden wie in red_band    
#         for band in bands_path:
                                   
#             band_name = band.split("_")[-1].split(".")[0]
#             print("Start with band: ", band_name)

#             band_arr = self.read_band(band_name)
#             band_arr_norm = self.normalize_band(band_arr, band_name, norm_textfile)
#             bands_patches[band_name] = self.patchify_band(band_arr_norm, patch_size)

#         if training:
            
#             outDir_patches =  os.path.join(out_dir, "training")
#             outDir_img = os.path.join(outDir_patches, "img")
#             outDir_mask = os.path.join(outDir_patches, "mask")
            
#             os.mkdir(outDir_patches)
#             os.mkdir(outDir_img)
#             os.mkdir(outDir_mask)
            
#             b2_path = self.get_band_path("B2")
#             savePatchesPV(bands_patches, outDir_img, outDir_mask, b2_path) # save patches of entire sentinel 2 tile for prediction 
#             print("Saved patches for training in folder: ", outDir_patches)
        
#         else:
            
#             outDir_patches = os.path.join(out_dir, "prediction")
#             os.mkdir(outDir_patches)
            
#             savePatchesFullImg(bands_patches, outDir_patches) # save patches of entire sentinel 2 tile for prediction 
#             print("Saved patches for prediction in folder: ", outDir_patches)
        
#         return outDir_patches
            
class Sentinel2(SatelliteImage):
    
    def __init__(self, tile_id, month, year):
        super().__init__(tile_id, month, year)
        self.satellite = "Sentinel-2"
        self.indizes_path = []
        self.tile_folder = f"/codede/{self.satellite}/MSI/L3-WASP/{self.year}/{self.month:02d}/01/S2_L3_WASP_{self.date_to_string()}_{self.tile_id}"       

    def calculate_indizes(self, out_dir, path=True):
        red = self.read_band("B4")
        nir = self.read_band("B8")
        swir1 = self.read_band("B11")
        ndvi = np.nan_to_num((nir-red)/(nir+red))
        ndwi = np.nan_to_num((nir-swir1)/(nir+swir1))
                
        # Get affine transformation and CRS from the original raster (e.g., "B4")
        with rasterio.open(self.get_band_path("B4")) as src:
            transform = src.transform
            crs = src.crs

        # Define rasterio profile for the output files
        profile = {
            'driver': 'GTiff',
            'height': ndvi.shape[0],
            'width': ndvi.shape[1],
            'count': 1,
            'dtype': ndvi.dtype,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
            'nodata': None
        }

        # Save NDVI and NDWI files with the affine transformation and CRS
        ndvi_path = os.path.join(out_dir, f"{self.tile_id}_NDVI.tif")
        ndwi_path = os.path.join(out_dir, f"{self.tile_id}_NDWI.tif")

        with rasterio.open(ndvi_path, 'w', **profile) as dst:
            dst.write(ndvi[:,:,0], 1)

        with rasterio.open(ndwi_path, 'w', **profile) as dst:
            dst.write(ndwi[:,:,0], 1)

        self.indizes_path = [ndvi_path, ndwi_path]
        print("Calculated Sentinel-2 indizes")
    
        if path:
            return [ndvi_path, ndwi_path]
        else: 
            return 
        
class Sentinel1(SatelliteImage):
    
    def __init__(self, tile_id, month, year):
        super().__init__(tile_id, month, year)
        self.satellite = "Sentinel-1"
        self.indizes_path = []
        self.tile_folder = f"/codede/{self.satellite}/SAR/CARD-BS-MC/{self.year}/{self.month:02d}/01/S1_CARD-BS-MC_{self.date_to_string()}_{self.tile_id}" 

    def calculate_indizes(self, out_dir, path=True):
        vv = self.read_band("VV")
        vh = self.read_band("VH")
        cr = np.nan_to_num(vh-vv)

        # Get affine transformation and CRS from the original raster (e.g., "B4")
        with rasterio.open(self.get_band_path("VV")) as src:
            transform = src.transform
            crs = src.crs

        # Define rasterio profile for the output files
        profile = {
            'driver': 'GTiff',
            'height': cr.shape[0],
            'width': cr.shape[1],
            'count': 1,
            'dtype': cr.dtype,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
            'nodata': None
        }

        # Save NDVI and NDWI files with the affine transformation and CRS
        cr_path = os.path.join(out_dir, f"{self.tile_id}_CR.tif")
       
        with rasterio.open(cr_path, 'w', **profile) as dst:
            dst.write(cr[:,:,0], 1)

        self.indizes_path = [cr_path]
        print("Calculated Sentinel-1 indizes")
    
        if path:
            return cr_path
        else: 
            return 

class Sentinel12(Sentinel1, Sentinel2):
    
    def __init__(self, tile_id, month, year):
        super().__init__(tile_id, month, year)
        self.s1 = Sentinel1(tile_id, month, year)
        self.s2 = Sentinel2(tile_id, month, year)
        self.indizes_path = []

    def get_bands_path(self):
        
        s1_bands = self.s1.get_bands_path()
        s2_bands = self.s2.get_bands_path()
        
        # Combine the bands
        combined_bands = s1_bands + s2_bands
        combined_bands.sort()
        
        return combined_bands
    
    def calculate_indizes(self, out_dir, path=True):
        
        s1_idx = self.s1.calculate_indizes(out_dir)
        s2_idx = self.s2.calculate_indizes(out_dir)
        
        # Combine the idx as list
        combined_idx = [s1_idx] + s2_idx
        combined_idx.sort()
        self.indizes_path = combined_idx
        
        print("Calculated Sentinel-1/-2 indizes")
    
        if path:
            return combined_idx
        else: 
            return 

norm_textfile = "/home/hoehn/data/output_model_data/Sentinel-12/normParameter.txt"
out_dir = "/home/hoehn/data/output_model_data/test"

S2_32UNA = Sentinel2("32UNA", 5, 2021)
S2_32UNA.initiate_bands()
# b2 = S2_32UNA.read_band("B2")
# b2_norm = S2_32UNA.normalize_band(b2, "B2", norm_textfile)
# S2_32UNA.plot_band(b2, "B2")
# S2_32UNA.plot_band(b2_norm, "B2_norm")

import pdb 
pdb.set_trace()

# S1_32UNA = Sentinel1("32UNA", 5, 2021)
# S12_32UNA = Sentinel12("32UNA", 5, 2021)
# S12_32UNA.calculate_indizes(out_dir, path=False)
# S12_32UNA.preprocess_tile(128, norm_textfile, out_dir)

# Save patches in different folders for training 
# use this structure for training 
# implement functions for training`?` -> or transfer this structure 
# plotting functions -> plot histogram of band 