import cv2
import json
import numpy as np
from pprint import pprint
import glob
import os

def draw_mask(json_filename, image_filename, resultant_filename):
    with open(json_filename) as data_file:
        data = json.load(data_file)
    # pprint(data)
    image = cv2.imread(image_filename)
    # image.shape
    mask = np.zeros((image.shape[0],image.shape[1],1), dtype=np.uint8)
    # mask = np.zeros((512,512,1), dtype=np.uint8)
    objs = data['shapes']
    color = [255,255,255]
    for obj in objs:
        points = np.array(list(obj['points']), dtype=np.int32)

        print(points)
        # print(color)
        if(points!=[]):
            cv2.fillPoly(mask, [points], color)
    # cv2.imshow("",mask)
    # cv2.waitKey(0)

    cv2.imwrite(os.path.join('lamlai_mask', resultant_filename.split('.')[0] + '.jpg'), mask)
    # cv2.imwrite('tagged/images'+image_filename, image)
    print('Image saved', resultant_filename)
def draw_black_images(file_images):
    for file_image in glob.glob(file_images + "\*.jpg"):
        image = cv2.imread(file_image)
        mask = np.zeros((image.shape[0],image.shape[1],1), dtype=np.uint8)
        cv2.imwrite(os.path.join('masks', os.path.basename(file_image)), mask)

for filename in os.listdir("json"):
    img_filename = os.path.join('F:/3k/images', filename.split('.')[0] + '.jpg')
    draw_mask(os.path.join('json',filename), img_filename, filename)

# draw_black_images("smoke_5_ko_khoi")