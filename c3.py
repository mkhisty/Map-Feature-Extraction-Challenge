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
import random as rand
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch
import torchvision
from torchvision.datasets import MNIST

os.makedirs(r'C:/Users/Owner/Desktop/USGS/Results3', exist_ok=True)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
"""def template():

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
        #plt.imshow(im)
        #plt.show()
    #—————————————————————————————————————————————————
 
    #ITRATING THROUGH SHAPES
    shape = data['shapes'][0]
    # read labels and bounding box coordinates
    label = shape['label']
    points = shape['points']
    xy_min, xy_max = points
    x_min, y_min = xy_min
    x_max, y_max = xy_max
           
    temp = im[int(y_min):int(y_max), int(x_min):int(x_max)]
    return temp"""
def bootstrapper(t,s,s2):
    s=s+1
    s2=s2+1
 
    x = np.shape(t)[1]
    y = np.shape(t)[0]
    r = np.zeros((y,x))
    maxx = 0
    maxy= 0
    minx=100000
    miny=100000
    print("x,Y:")
    print(x,y)
    for i in range(0,y):
        for j in range(0,x):
            if(np.average(t[i][j])<120):
                r[i][j]=1
                if(j>maxx):
                    #print("max:",j)
                    maxx = j
                if(i>maxy):
                    #print("maxi:",i)
                    maxy = i
                if(j<minx):
                    #print("minx:",j)
                    minx = j
                if(i<miny):
                    #print(i)
                    miny = i
    plt.imshow(t)
    plt.show()
    plt.imshow(r)
    plt.show()
    
    right = []
    wrong=[]
    yb_1 = y-maxy
    yb_2 = miny-1
    xb_1 = x-maxy
    xb_2 = minx
    yb = yb_1+yb_2
    xb = xb_1+xb_2
    t2 =[i[minx:maxx+1] for i in r[miny:maxy+1]]
    plt.imshow(t2)
    plt.show()
    for k in range(0,s):
        t7 = np.asarray(t2)
        noise = np.zeros(t7.shape)
        b1 = 65
        b2 = 83
        for i in range(t7.shape[0]):
            for j in range(t7.shape[1]):
                ri = rand.randint(1,100)
                if(ri>b1 and ri<b2):
                    if(t7[i][j]==1):
                        noise[i][j]=-1
                elif(ri>b2):
                    if(t7[i][j]==0):
                        noise[i][j]=1
        t8 = t7+noise

        #plt.imshow(t8)
        #plt.show()
        tc = t8.tolist()[:]
        ybf1 = rand.randint(0,yb)
        ybf2 = yb-ybf1
        xbf1 = rand.randint(0,xb)
        xbf2 = xb-xbf1
        #print("y",ybf1,ybf2)
        #print(y,maxy,miny)
        #tc.append([0 for i in range(0,x)] for j in range(0,ybf1))
        p=1
        #tc = [[0 for j in range(0,minx)].extend(tc[i]).extend([0 for k in range(0,maxx)]) for i in range(0,y)]
        x1 = [0 for j in range(0,xbf2)]
        x2 = [0 for j in range(0,xbf1)]
        fr = [0 for j in range(0,x)]
        y1 = [fr for i in range(0,ybf1)]
        y2 = [fr for i in range(0,ybf2)]
        #print(y1,y2)
        tn = []
        plc1 = x1
        for i in tc:
            plc1 = x1[:]
            #print(len(plc1))
            plc1.extend(i[:])
            plc1.extend(x2[:])
            tn.append(plc1)
            #print(len(plc1))
            del plc1
        #print(len(tn),len(tn[0]),xbf1+xbf2)
        #plt.imshow(tn)
        #plt.show()
 
        if y2:
            tn.extend(y2)
        if y1:
            y1.extend(tn)
        if y2 and (not y1):
            right.append(tn)
        else:
            right.append(y1)
    for i in range(0,s2):
        #w = np.zeros(r.shape, dtype='i8')
        w = [[0 for i in range(0,r.shape[1])] for j in range(0,r.shape[0])]
        b3 = 50
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                ri = rand.randint(1,100)
                if(ri>b3):
                        w[i][j]=1
 
        #plt.imshow(w)
        #splt.show()
        wrong.append(w)
    dataset = []
    y = []
    for i in right:
        dataset.append([i,1])
    for j in wrong:
        dataset.append([j,0])
    #print(type(dataset),type(dataset[1]),type(dataset[1][0]),type(dataset[1][0][0]),type(dataset[1][0][0][0]))
    return dataset
 
"""t = template()
""""""

w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    img = dataset[i][0]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    img = dataset[110+i][0]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
"""


def ml(dataset,t):

    """print("DATASET:",len(dataset),len(dataset[0]),len(dataset[0][0]),len(dataset[0][0][0]),dataset[0][0][0][0])
    """
    train_ds, val_ds = random_split(dataset, [102, 50])
    batch_size = 2
    d2 = np.array(dataset)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size,shuffle=True)
    x = np.shape(t)[1]
    y = np.shape(t)[0]
    input_size = y*x
    num_classes = 2

    # Logistic regression models
    model = nn.Linear(input_size, num_classes)

    class MnistModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_size, num_classes)
            
        def forward(self, xb):
            xb = xb.reshape(-1, 784)
            out = self.linear(xb)
            return out
        
    model = MnistModel()
    return model
"""   for images, labels in train_loader:
        print(images[0][0])
        print(len(images),len(images[0]),len(images[0][0]))
        i = np.array(images, dtype="int64")[0]
        i2 = torch.from_numpy(i)
        outputs = model(i2)
        print(outputs)
        break"""
        

def predict(i,m):
    return 1
    #Would ideally run input through model

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
