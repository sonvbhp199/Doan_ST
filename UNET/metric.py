
import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from model import build_unet
from utils import create_dir, seeding
t=0
def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1
    return(TN, FP, FN, TP)
def metric(y_true,y_pred):
    global t
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred,labels=[0,1]).ravel()
    
    tn, fp, fn, tp = perf_measure(y_true,y_pred)
    if (tp+fp+fn)==0:
        t +=1
        iou = 1
    else: 
        iou = tp/(tp+fp+fn)
    return [iou, fp/(fp+tn)]
    # return [jaccard_score(y_true,y_pred),0.0]
def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)
    
    return metric(y_true,y_pred)


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    # create_dir("results")

    """ Load dataset """
    test_x = sorted(glob("D:/dataset/smoke/data_unet_512/test/images/*"))
    test_y = sorted(glob("D:/dataset/smoke/data_unet_512/test/masks/*"))
    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    metrics_score = [0.0,0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        # name = x.split("/")[-1].split(".")[0]
        name =os.path.basename(x)
        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))     ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        ## mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)
        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)
            score = calculate_metrics(y, pred_y)
            if score[0]<0.7:
                with open('files/iou.txt', mode='a') as f:
                    f.write(name+":"+ str(score[0])+"\n")
            metrics_score = list(map(add, metrics_score, score))


    print(t)
    iou = metrics_score[0]/(len(test_x))
    far = metrics_score[1]/len(test_x)
    print(f"iou: {iou:1.4f} - far: {far:1.4f}")

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)
