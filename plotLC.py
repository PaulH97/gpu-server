import matplotlib.pyplot as plt 
import numpy as np
# import pandas as pd

# loss_train = pd.read_csv("/home/hoehn/data/output/Sentinel-2/T1_train_loss.csv")
# loss_val = pd.read_csv("/home/hoehn/data/output/Sentinel-2/T1_validation_loss.csv")

# recall_train = pd.read_csv("/home/hoehn/data/output/Sentinel-2/T1_train_recall.csv")
# recall_val = pd.read_csv("/home/hoehn/data/output/Sentinel-2/T1_validation_recall.csv")

# plt.figure(figsize=(14,6))
# plt.subplot(121)
# plt.plot(loss_train["Step"], loss_train["Value"], label="Train", color="royalblue")
# plt.plot(loss_val["Step"], loss_val["Value"],  label="Validation", color="darkorange")
# plt.title("Learning Curves of Model Metric: Loss", fontsize = 14)
# plt.ylabel('Loss', fontsize = 10)
# plt.xlabel('Epochs', fontsize = 10)
# plt.legend()
# plt.subplot(122)
# plt.plot(recall_train["Step"], recall_train["Value"], label="Train", color="royalblue")
# plt.plot(recall_val["Step"], recall_val["Value"],  label="Validation", color="darkorange")
# plt.title("Learning Curves of Model Metric: Recall", fontsize = 14)
# plt.ylabel('Recall', fontsize = 10)
# plt.xlabel('Epochs', fontsize = 10)
# plt.legend()
# plt.savefig("T1_loss_recall.png")

# from tools_preprocess import rasterizeShapefile
# tile_name = "32UPE"
# output_folder = "/home/hoehn/data/output/Sentinel-2/prediction/32UPE"
# solar_path = "/home/hoehn/data/input/solarParks_testTiles.shp"
# raster_muster = "/codede/Sentinel-2/MSI/L3-WASP/2021/06/01/S2_L3_WASP_202106_32UPE/SENTINEL2X_20210615-000000-000_L3A_T32UPE_C_V1-2_FRC_B2.tif"
# mask_path = rasterizeShapefile(raster_muster, solar_path, output_folder, tile_name, col_name="SolarPark")

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

models = ["128_idx", "128_noidx", "256_idx", "256_noidx"]

# s2_recall = [0.857,	0.85,	0.838,	0.845]		
# s2_precision = [0.907,	0.905,	0.887,	0.902]
# s2_fscore = [0.881,	0.877,	0.862,	0.872]
# s2_iou = [0.892, 0.888, 0.878, 0.886]

# s12_recall = [0.863,	0.856,	0.835,	0.841]	
# s12_precision = [0.911,	0.908,	0.908,	0.9]
# s12_fscore = [0.886,	0.881,	0.87,	0.869]
# s12_iou = [0.896,	0.892,	0.884,	0.884]	

s2_recall = [0.9018, 0.8808, 0.8628, 0.8823]
s12_recall = [0.8930, 0.8831, 0.8812, 0.8801]

s2_precision = [0.7278, 0.7709, 0.8407, 0.8290]
s12_precision = [0.7651, 0.8236, 0.7856, 0.7971]

s2_fscore = [0.8055,0.8222,0.8516,0.8549]
s12_fscore = [0.8241,0.8523,0.8307,0.8365]

s2_iou = [0.6744,0.6981,0.7416,0.7465]
s12_iou = [0.7008,0.7427,0.7104,0.7190]

# Set position of bar on X axis
br1 = np.arange(len(s2_recall))
br2 = [x + barWidth for x in br1]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False)
fig.suptitle('Model Metrics of K-Fold Cross-Validation', fontsize=20, weight="bold")
ax1.bar(br1, s2_recall, color ='mediumspringgreen', width = barWidth,
        edgecolor ='grey', label ='Sentinel-2')
ax1.bar(br2, s12_recall, color ='mediumpurple', width = barWidth,
        edgecolor ='grey', label ='Sentinel-1/-2')
ax1.set_title("Recall", weight = "bold")
ax1.set_ylim(0.60, 0.95)
ax1.set_xticks([r +barWidth/2 for r in range(len(s2_recall))], models)
ax1.set_ylabel("Metric", fontsize=13, labelpad=15)
ax1.legend()
ax2.bar(br1, s2_precision, color ='mediumspringgreen', width = barWidth,
        edgecolor ='grey', label ='Sentinel-2')
ax2.bar(br2, s12_precision, color ='mediumpurple', width = barWidth,
        edgecolor ='grey', label ='Sentinel-1/-2')
ax2.set_title("Precision", weight = "bold")
ax2.set_ylim(0.60, 0.95)
ax2.set_xticks([r +barWidth/2 for r in range(len(s2_recall))], models)
ax2.legend()
ax3.bar(br1, s2_fscore, color ='mediumspringgreen', width = barWidth,
        edgecolor ='grey', label ='Sentinel-2')
ax3.bar(br2, s12_fscore, color ='mediumpurple', width = barWidth,
        edgecolor ='grey', label ='Sentinel-1/-2')
ax3.set_title("F1-Score", weight = "bold")
ax3.set_ylim(0.60, 0.95)
ax3.set_xticks([r +barWidth/2 for r in range(len(s2_recall))], models)
ax3.set_xlabel("Models", fontsize=13, labelpad=15)
ax3.set_ylabel("Metric", fontsize=13, labelpad=15)
ax3.legend()
ax4.bar(br1, s2_iou, color ='mediumspringgreen', width = barWidth,
        edgecolor ='grey', label ='Sentinel-2')
ax4.bar(br2, s12_iou, color ='mediumpurple', width = barWidth,
        edgecolor ='grey', label ='Sentinel-1/-2')
ax4.set_title("IoU", weight = "bold")
ax4.set_ylim(0.60, 0.95)
ax4.set_xticks([r +barWidth/2 for r in range(len(s2_recall))], models)
ax4.set_xlabel("Models", fontsize=13, labelpad=15)
ax4.legend()
plt.gcf().set_dpi(600)
plt.savefig("barplot_32uqv.png")

# from tools_preprocess import load_img_as_array
# sen2_img = load_img_as_array("/home/hoehn/data/output/Sentinel-2/crops128/idx/train/img/33UVT_img_5971_pv.tif")
# sen2_mask = load_img_as_array("/home/hoehn/data/output/Sentinel-2/crops128/idx/train/mask/33UVT_mask_5971_pv.tif")
# sen1_img = load_img_as_array("/home/hoehn/data/output/Sentinel-1/crops/idx/train/img/33UVT_img_5971_pv.tif")
# sen1_mask = load_img_as_array("/home/hoehn/data/output/Sentinel-1/crops/idx/train/mask/33UVT_mask_5971_pv.tif")

# bgr = sen2_img[:,:,2:5] 
# rgb = np.dstack((bgr[:,:,2],bgr[:,:,1],bgr[:,:,0]))
# vv = sen1_img[:,:,1]
# vh = sen1_img[:,:,0]

# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
# ax1.set_title("RGB Image", fontsize=12)
# ax1.imshow(rgb) 
# ax2.set_title("Mask of RGB Image", fontsize=12)
# ax2.imshow(sen2_mask)
# ax3.set_title("VV Image", fontsize=12)
# ax3.imshow(vv) 
# ax4.set_title("Mask of VV Image", fontsize=12)
# ax4.imshow(sen1_mask)
# ax5.set_title("VH Image", fontsize=12)
# ax5.imshow(vh) 
# ax6.set_title("Mask of VH Image", fontsize=12)
# ax6.imshow(sen1_mask)

# plt.gcf().set_dpi(600)
# plt.savefig("train_patches.png")

# dirs = os.listdir("/home/hoehn/data/output/Sentinel-2/indizes/")
# dirs = sorted(dirs)

# for band in dirs:

#     band = os.path.join("/home/hoehn/data/output/Sentinel-2/indizes/", band)
#     band_name = os.path.basename(band).split(".")[0].split("_")[-1]
#     tile_name = os.path.basename(band).split(".")[0].split("_")[0]
#     raster = rasterio.open(band)
#     r_array = raster.read()[:,:10980,:10980]
#     r_array = np.expand_dims(r_array, axis=0)
#     r_array = np.moveaxis(r_array, 0, -1)
#     r_array = np.nan_to_num(r_array)
    
#     print("Tile ID: ", tile_name)
#     print("Band: ", band_name)

#     bands_scale = json.load(open("/home/hoehn/data/output/Sentinel-2/normParameter.txt"))

#     # 1 and 99 perzentile + [0,1]            
#     a,b = 0,1
#     c,d = bands_scale[band_name.upper()]
#     r_array_norm = (b-a)*((r_array-c)/(d-c))+a
#     r_array_norm[r_array_norm > 1] = 1
#     r_array_norm[r_array_norm < 0] = 0           
#     #tiff.imwrite(f'{band}_norm.tif', r_array_norm)
   
#     print("Creating plot")
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
#     ax1.set_title("Original Histo", fontsize=12)
#     ax1.hist(r_array.flatten(), bins=100) 
#     ax2.set_title("Normalized Histo", fontsize=12)
#     ax2.hist(r_array_norm.flatten(), bins=100)
#     plt.savefig(f"{tile_name}_histo2_{band_name}.png")

# import rasterio
# import json

# band = "/codede/Sentinel-2/MSI/L3-WASP/2021/06/01/S2_L3_WASP_202106_32ULB/SENTINEL2X_20210615-000000-000_L3A_T32ULB_C_V1-2_FRC_B11.tif"
# band_name = "B11"
# raster = rasterio.open(band)
# r_array = raster.read()[:,:10980,:10980]
# r_array = np.expand_dims(r_array, axis=0)
# r_array = np.moveaxis(r_array, 0, -1)
# r_array = np.nan_to_num(r_array)

# # Min Max Scaling 
# r_array_flat = r_array.flatten()
# r_array_minmax = ((r_array-np.amin(r_array_flat))/(np.amax(r_array_flat)-np.amin(r_array_flat)))

# # 1 and 99 perzentile 
# f,l = np.nanpercentile(r_array, [1,99])
# r_array_1_99 = ((r_array-f)/(l-f))

# bands_scale = json.load(open("/home/hoehn/data/output/Sentinel-2/normParameter.txt"))

# # 1 and 99 perzentile + [0,1]            
# a,b = 0,1
# c,d = bands_scale[band_name.upper()]
# r_array_norm = (b-a)*((r_array-c)/(d-c))+a
# r_array_norm[r_array_norm > 1] = 1
# r_array_norm[r_array_norm < 0] = 0     

# # Plot histogram of linear normalization 
# q25, q75 = np.percentile(r_array, [25, 75])
# bin_width = 2 * (q75 - q25) * len(r_array) ** (-1/3)
# bins = round((r_array.max() - r_array.min()) / bin_width)   
# bins = 150

# rows, cols = 2, 2
# plt.figure(figsize=(20,20))
# plt.subplot(rows, cols, 1)
# plt.title("Band {} - Original Histogram".format(band_name), fontsize = 20)
# plt.hist(r_array.flatten(), bins = bins, color="lightcoral")
# plt.ylabel('Number of Pixels', fontsize = 16)
# plt.xlabel('DN', fontsize = 16)
# plt.subplot(rows, cols, 2)
# plt.title("Band {} - MinMax Normalization".format(band_name), fontsize = 20)
# plt.hist(r_array_minmax.flatten(), bins = bins, color="lightblue")
# plt.ylabel('Number of pixels', fontsize = 16)
# plt.xlabel('Normalized values', fontsize = 16)
# plt.subplot(rows, cols, 3)
# plt.title("Band {} - Normalization 1st/99th Percentile".format(band_name), fontsize = 20)
# plt.hist(r_array_1_99.flatten(), bins = bins, color="lightblue")
# plt.ylabel('Number of pixels', fontsize = 16)
# plt.xlabel('Normalized values', fontsize = 16)
# plt.subplot(rows, cols, 4)
# plt.title("Band {} - Normalization 1st/99th Percentile [0,1]".format(band_name), fontsize = 20)
# plt.hist(r_array_norm.flatten(), bins = bins, color="lightblue")
# plt.ylabel('Number of pixels', fontsize = 16)
# plt.xlabel('Normalized values', fontsize = 16)
# plt.gcf().set_dpi(600)
# plt.savefig(f"HistoB11.png")
