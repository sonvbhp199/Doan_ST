from matplotlib import image
import numpy as np
import cv2
import os
import json
from sklearn.model_selection import train_test_split
idx = 0
img = os.listdir("data_unet/test/images")
# train,test = train_test_split(img,test_size=0.1,random_state=20)
path_images = 'D:/son/read_json/data_unet/test/images'
path_masks='D:/son/read_json/data_unet/test/masks'
dataset_dicts=[]
for file in img:
# for file in os.listdir(path_images):
#     ex = file.split('.')[1]
#     if not ex == 'png' and not ex == 'jpg':
#         print(f"{file} is not image")
#         continue
    frame_idx = file.split('.')[0]
    record = {}
    img_path = os.path.join(path_images,file)
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    record["file_name"] = file
    record["height"] = height
    record["width"] = width
    sset = "test"
    objs = []
    path_mask = os.path.join(path_masks,file)
    if os.path.exists(path_mask):
        mask = cv2.imread(path_mask)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _,contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            obj = {
                "bbox": [x, y, x+w, y+h],
                "segmentation": c.flatten().tolist(),
                "objects": "smoke",
            }
            objs.append(obj)
    record["image_id"] = idx
    idx += 1
    record["regions"] = objs
    dataset_dicts.append(record)
print(idx)
with open('data_mask-rcnn/test/test.json', 'w') as outfile:
    json.dump(dataset_dicts, outfile)