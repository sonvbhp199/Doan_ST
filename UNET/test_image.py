import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from model import build_unet

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask
def pred_smoke(model, image_path):
    # name =image_path.split("/")[1]
    name =os.path.basename(image_path)
    H = 512
    W = 512
    size = (W, H)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) ## (512, 512, 3)
    image = cv2.resize(image, size)
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
    # line = np.ones((size[1], 10, 3)) * 128
    # cat_images = np.concatenate([image, line, pred_y, line, pre_somke], axis=1)
    # cv2.imwrite(f"result3/{name}", cat_images)

    cv2.imshow("",pre_somke)
    cv2.waitKey(0)
if __name__ == "__main__":
    # test_x = sorted(glob("D:/dataset/smoke/data_unet_512/test/images/*"))
    checkpoint_path = "files/checkpoint.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # for image_path in test_x:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = build_unet()
    #     model = model.to(device)
    #     checkpoint = torch.load(checkpoint_path,map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     model.eval()
    #     pred_smoke(model,image_path)
    path = 'img/1.jpg'
    pred_smoke(model, path)
   

