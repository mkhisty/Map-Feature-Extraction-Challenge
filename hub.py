import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
import os
from joblib import Parallel, delayed
import math
import json
from datetime import datetime
from scipy import sparse
from PIL import Image
import requests
import shutil
import rasterio
import warnings
import metric2
import b2 as baseline
import c3
#file_name = r'AZ_Arivaca_314329_1941_62500_geo_mosaic'
#file_loc = r'C:/Users/Owner/Desktop/USGS/Data/Training/training/'
file_loc = r'C:/Users/Owner/Desktop/USGS/Data/Training/training/'
results_loc = r'C:/Users/Owner/Desktop/USGS/Results'
#feature= r"3_pt"
finalr2=[]
directory = os.fsencode(file_loc)
for file in os.listdir(directory):
    p = str(os.path.basename(file))
    ext = p.split(".")
    file_name=ext[0][2:]
    if(ext[1]=="tif'"):
        src = rasterio.open(file_loc+file_name+r'.tif')
        #if(src.count==3):
        print("working on file"+src.name)
        featurelist= baseline.predict(file_loc,file_name+r'.tif')
        finalr=[]
        #for i in featurelist:
            #feature = i['label']
            #finalr.append(metric2.feature_f_score(file_loc+file_name+r'.tif',results_loc+file_name+r'_'+feature+r'.tif',file_loc+file_name+r'_'+feature+r'.tif'))
            #im=cv2.imread(file_loc+file_name+r'_'+feature+r'.tif')
            #im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #arr = np.asarray(im)
            #loc = np.where(arr==1)
            #print("LOC: \n:",loc)
        #finalr2.append(finalr)
print(finalr2)




#results = metric2.feature_f_score(r'C:/Users/Owner/Desktop/USGS/Data/Training/training/AZ_Arivaca_314329_1941_62500_geo_mosaic.tif',r'C:/Users/Owner/Desktop/USGS/Results/AZ_Arivaca_314329_1941_62500_geo_mosaici3_pt.tif',r'C:/Users/Owner/Desktop/USGS/Data/Training/training/AZ_Arivaca_314329_1941_62500_geo_mosaic_3_pt.tif')
