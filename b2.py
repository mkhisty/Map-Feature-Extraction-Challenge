import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import requests
import shutil
import os
import rasterio
import warnings
import sys
import pprint
import c3

os.makedirs(r'C:/Users/Owner/Desktop/USGS/Results3', exist_ok=True)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

data_dir=r'C:/Users/Owner/Desktop/USGS/Data/Training/training'
file_name = r'AZ_Arivaca_314329_1941_62500_geo_mosaic.tif'
#for file_name in os.listdir(data_dir):(WHEN ITERATING)
    # get the .tif files

#PROCESSING FILES
if '.tif' in file_name:
    filename=file_name.replace('.tif', '')
    print('Working on map:', file_name)
    file_path=os.path.join(data_dir, file_name)
    test_json=file_path.replace('.tif', '.json')

    # read the legend annotation file
    with open(test_json) as f:
        data = json.load(f)

    # load image into an array
    im=cv2.imread(file_path)
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.show()
#—————————————————————————————————————————————————

#ITRATING THROUGH SHAPES
    for shape in data['shapes']:
        # read labels and bounding box coordinates
        label = shape['label']
        points = shape['points']
        xy_min, xy_max = points
        x_min, y_min = xy_min
        x_max, y_max = xy_max
        
        template = im[int(y_min):int(y_max), int(x_min):int(x_max)]
        h, w = template.shape[0], template.shape[1]
        print('using the following legend feature for matching...:')
        plt.imshow(template)
        plt.show()
        
        print('detecting for label:', label)
        typ=label.split('_')[-1]
        print('type:', typ)
        
        np.set_printoptions(threshold=sys.maxsize)
        print(template)
        print(np.shape(template))

        ## To match point shapes
        if typ=='pt':
            
            # find all the template matches in the basemap
            res = cv2.matchTemplate(im, template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.3
            loc = np.where( res >= threshold)
            dataset= c3.bootstrapper(template,100,50)
            m = c3.ml(dataset)
            # use the bounding boxes to create prediction binary raster
            pred_binary_raster=np.zeros((im.shape[0], im.shape[1]))
            for pt in zip(*loc[::-1]):
                if(c3.predict(im[pt[1]:pt[1] + h, pt[0]:pt[0] + w],m)):
                    print('match found:')
                    
                    pred_binary_raster[int(pt[1]+float(h)/2), pt[0] + int(float(w)/2)]=1
                    plt.imshow(im[pt[1]:pt[1] + h, pt[0]:pt[0] + w])
                    plt.show()

        
        ## To match lines and polygons
        else:
            
            if typ=='line':
                # do edge detection
                print('detecting lines in the legend feature...')
                gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, threshold1=30, threshold2=100)
                plt.imshow(edges)
                plt.show()
                central_pixel=tuple(np.argwhere(edges==255)[0])
                sought = template[central_pixel].tolist()
            else: # type=='poly'
                # take the median of the colors to find the predominant color
                r=int(np.median(template[:,:,0]))
                g=int(np.median(template[:,:,1]))
                b=int(np.median(template[:,:,2]))
                sought=[r, g, b]
            
            print('matching the color:', sought)
            
            lower = np.array(sought)-4
            lower[lower<0] = 0
            lower=tuple(lower.tolist())
            upper = np.array(sought)+4
            upper[upper>255] = 255
            upper=tuple(upper.tolist())
            pred_binary_raster = cv2.inRange(im, lower, upper)/255
        
        # print
        print('predicted binary raster:')
        print('shape:', pred_binary_raster.shape)
        print('unique value(s):', np.unique(pred_binary_raster))

        # plot the raster and save it
        plt.imshow(pred_binary_raster)
        plt.show()
        
        # save the raster into a .tif file
        out_file_path=os.path.join(r'C:\Users\Owner\Desktop\USGS\Results3', filename+'_'+label+'.tif')
        out_file_path2=os.path.join(r'C:\Users\Owner\Desktop\USGS\Results3', filename+'i'+label+'_.tif')

        pred_binary_raster=pred_binary_raster.astype('uint16')
        cv2.imwrite(out_file_path, pred_binary_raster)
        
        # convert the image to a binary raster .tif
        raster = rasterio.open(out_file_path)
        array = raster.read(1)
        with rasterio.Env():
            profile = raster.profile
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw')
            with rasterio.open(out_file_path2, 'w', **profile) as dst:
                dst.write(array.astype(rasterio.uint8), 1)
        os.remove(out_file_path2) 
    
        

# %%
# Test; load and plot the produced rasters
    for file_name in os.listdir('results'):
        file_path = os.path.join('results', file_name)
        #print('file_path:', file_path)
        im=cv2.imread(file_path)
        #print(np.unique(im), im.shape)
        im[np.where(im==1)]=255
        #plt.imshow(im)
        #plt.show()
    

# %%



