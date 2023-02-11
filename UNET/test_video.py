import os, time
from operator import add
from tkinter.tix import IMAGE
from unittest import result
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding
fps = 0
def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask
def pred_smoke(model, image):
    # name =image_path.split("/")[1]
    # name =os.path.basename(image_path)
    # H = 512
    # W = 512
    # size = (W, H)
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR) ## (512, 512, 3)
    # image = cv2.resize(image, size)
    x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
        pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)
    pred_y = mask_parse(pred_y)
    pred_y = np.concatenate([pred_y * 255], axis=1)
    smoke = np.copy(image)
    smoke[(pred_y==255).all(-1)] = [0,255,0]
    pre_somke = cv2.addWeighted(smoke, 0.3, image, 0.7, 0, smoke)
    return pre_somke
def pred_gen(src):
    if src.endswith(('mp4', 'avi')):
        vicap = cv2.VideoCapture(src)
        print(vicap.get(cv2.CAP_PROP_FPS))
        global fps 
        fps = vicap.get(cv2.CAP_PROP_FPS) 
        ret,image = vicap.read()
        while ret:
            ret, image = vicap.read()
            if(ret==True):
                image = cv2.resize(image, (512, 512), cv2.IMREAD_COLOR)
                yield image
def make_video(src):
    global fps
    images = [img for img in pred_gen(src)]
    frame = images[0]
    height, width, layers = frame.shape  
    video = cv2.VideoWriter('pred.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps , (width,height))
    i=0
    for image in images:
        result= pred_smoke(model, image)
        video.write(result)
    cv2.destroyAllWindows() 
    video.release()
    print(fps)
if __name__ == "__main__":
    checkpoint_path = "files/checkpoint.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # pred_gen("video_test.mp4")
    # print(fps)
    make_video("video_test.mp4")