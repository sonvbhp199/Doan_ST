# Import necessary libraries
from importlib.resources import path
from tkinter import image_names
import cv2
import numpy as np
import glob
import os
s = "lamlaimask"
d= "lamlaimask_512"
for file in os.listdir(s):
    mask = cv2.imread(os.path.join(s,file))
    t = cv2.resize(mask, (512, 512))
    cv2.imwrite(os.path.join(d,file),t)
