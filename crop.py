from skimage.data import imread
from skimage.io import imshow,imsave
from skimage import img_as_float
import pandas as pd
import numpy as np
import cv2
from skimage.util import crop
from skimage.transform import rotate
from skimage.transform import resize
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import math

def deg_angle_between(x1,y1,x2,y2):
    from math import atan2, degrees, pi
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    return(degs)

def get_rotated_cropped_fish(img,x1,y1,x2,y2):
    (h,w) = img.shape[:2]
    #calculate center and angle
    center = ( (x1+x2) / 2,(y1+y2) / 2)
    angle = np.floor(-deg_angle_between(x1,y1,x2,y2))
    #print('angle=' +str(angle) + ' ')
    #print('center=' +str(center))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    fish_length = np.sqrt((x1-x2)**2+(y1-y2)**2)
    print fish_length, center[0], center[1]
    cropped = rotated[(max((center[1]-fish_length/1.8),0)):(max((center[1]+fish_length/1.8),0)) ,
                      (max((center[0]- fish_length/1.8),0)):(max((center[0]+fish_length/1.8),0))]
    print ('success')
    # imshow(img)
    # imshow(rotated)
    # imshow(cropped)
    resized = resize(cropped,(224,224))
    return(resized)


label_files = ['/home/vinod/kaggle/fisheries_monitor/croping/train/labels/bet_labels.json',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/labels/alb_labels.json',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/labels/yft_labels.json',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/labels/dol_labels.json',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/labels/shark_labels.json',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/labels/lag_labels.json',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/labels/other_labels.json']

data_dirs = ['/home/vinod/kaggle/fisheries_monitor/croping/train/BET/',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/ALB/',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/YFT/',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/DOL/',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/SHARK/',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/LAG/',
             '/home/vinod/kaggle/fisheries_monitor/croping/train/OTHER/']


images = list()
labels_list = list()
for c in range(7):
    labels = pd.read_json(label_files[c])
    for i in range(len(labels)):
        try:
            img_filename = labels.iloc[i,2]
            print(img_filename)
            l1 = pd.DataFrame((labels[labels.filename==img_filename].annotations).iloc[0])
            image = imread(data_dirs[c]+img_filename)
            #images.append(get_rotated_cropped_fish(image,np.floor(l1.iloc[0,1]),np.floor(l1.iloc[0,2]),np.floor(l1.iloc[1,1]),np.floor(l1.iloc[1,2])))
            images = get_rotated_cropped_fish(image,np.floor(l1.iloc[0,1]),np.floor(l1.iloc[0,2]),np.floor(l1.iloc[1,1]),np.floor(l1.iloc[1,2]))
            labels_list.append(c)
            imsave(data_dirs[c]+'preprocessed_train/'+img_filename, images)
        except:
            pass

'''
for i in range(len(images)):
    imsave('preprocessed_train/img_'+str(i)+'label_'+str(labels_list[i])+'.jpg',images[i])
'''