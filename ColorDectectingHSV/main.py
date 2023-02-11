from glob import glob
from itertools import count
import cv2
import numpy as np
import matplotlib.pylab as plt
import glob
imgs= glob.glob("test/*.jpg")
img_path= imgs[1960:]
for img_path in img_path:
    print(img_path)
    image = cv2.imread(img_path,)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # height, width = image.shape[0:2]
    # img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # img_hsv = cv2.resize(image,(width,height))
    lower_range = np.array([0, 0, 109])
    upper_range = np.array([180, 255, 255])
    mask = cv2.inRange(img_hsv, lower_range, upper_range)
    height, width = mask.shape
    mask_find = np.zeros((height,width))
    _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  
    color = (255,255,255)
    if contours:
        c= max(contours,key=cv2.contourArea)
        cv2.fillPoly(mask_find, pts=[c], color=color)
    cv2.imwrite(img_path.replace("test","mask",1),mask_find)