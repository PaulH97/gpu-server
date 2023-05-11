import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import make_interp_spline

data = pd.read_csv("/home/hoehn/data/output/Sentinel-2/specSignature.csv")

year = {	'01':'January',
		'02':'February',
		'03':'March',
		'04':'April',
		'05':'May',
		'06':'June',
		'07':'July',
		'08':'August',
		'09':'September',
		'10':'October',
		'11':'November',
		'12':'December'		}


wavelength = [
    "0.490",
    "0.560",
    "0.665", 
    "0.705",
    "0.740",
    "0.783", 
    "0.842",	
    "0.865",
    "1.610",
    "2.190"	  
]

for month in range(1,13):

    if month != 4:
        if month < 10:
            month = "0" + str(month)
            year_month = "2021-" + month
        else: 
            month = str(month)
            year_month = "2021-" + month

        df = data[data["month"] == year_month]

        df.drop(columns=df.columns[0], axis=1,  inplace=True)
        df.drop(columns=df.columns[-1], axis=1,  inplace=True)
        df_t = df.transpose()
        df_t.index = sorted(df_t.index.values, key=lambda x: int("".join([i for i in x if i.isdigit()])))

        df_t["wavelength"] = wavelength
        col_names = list(df_t.columns)

        
        plt.plot(df_t["wavelength"], df_t[col_names[0]]/10000, marker=".",  label="Solar", color="grey")
        plt.plot(df_t["wavelength"], df_t[col_names[1]]/10000, marker=".",  label="Grassland", color="yellowgreen")
        plt.plot(df_t["wavelength"], df_t[col_names[2]]/10000, marker=".",  label="Forest", color="darkgreen")
        plt.plot(df_t["wavelength"], df_t[col_names[3]]/10000, marker=".",  label="Sealed Surfaces", color="maroon")
        plt.plot(df_t["wavelength"], df_t[col_names[4]]/10000, marker=".",  label="Cropland", color="goldenrod")
        plt.title("Spectral Signatures of Different Landuse in {}".format(year[month]), fontsize = 12)
        plt.ylabel('Reflectance in %', fontsize = 10)
        plt.xlabel('Wavelength (µm)', fontsize = 10)
        plt.legend()
        plt.show()
        plt.savefig("test_{}.png".format(month))
        plt.clf()




