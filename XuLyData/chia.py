import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
img = os.listdir("data_unet/train/images")
train,test = train_test_split(img,test_size=0.1,random_state=20)
# for file in train:
#     shutil.copy(os.path.join("images",file),"data_unet/train/images")
#     shutil.copy(os.path.join("masks",file),"data_unet/train/masks")
# for file in test:
#     shutil.copy(os.path.join("images", file), "data_unet/test/images")
#     shutil.copy(os.path.join("masks", file), "data_unet/test/masks")
for file in train:
    print()