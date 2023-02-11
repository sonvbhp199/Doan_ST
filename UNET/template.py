import streamlit as st
from PIL import Image
import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
import torch
from model import build_unet
import tempfile
fps=0
n_f =0
def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask
def pred_smoke(model, image):
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
    vicap = cv2.VideoCapture(src)
    global fps
    fps = vicap.get(cv2.CAP_PROP_FPS) 
    ret,image = vicap.read()
    while ret:
        ret, image = vicap.read()
        if(ret==True):
            image = cv2.resize(image, (512, 512), cv2.IMREAD_COLOR)
            yield image
def make_video(src):
    global fps,n_f
    images = [img for img in pred_gen(src)]
    n_f =len(images)
    frame = images[0]
    height, width, layers = frame.shape
    # file_out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    video = cv2.VideoWriter('pred.mp4',cv2.VideoWriter_fourcc(*'avc1'), fps , (width,height))
    i=0
    # dim = (512, 288)
    for image in images:
        # image = cv2.resize(image, (512, 512), cv2.IMREAD_COLOR)
        result= pred_smoke(model, image)
        # result =image
        video.write(result)
    cv2.destroyAllWindows() 
    video.release()


checkpoint_path = "files/checkpoint.pth"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = build_unet()
model = model.to(device)
checkpoint = torch.load(checkpoint_path,map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# st.header("Smoke segmentation")
st.markdown("<h1 style='text-align: center; color: black;'>Smoke segmentation</h1>", unsafe_allow_html=True)
file = st.file_uploader("choose a file", type=['jpg','mp4','avi'])
c1, c2= st.columns(2)
if file:    

    # image = Image.open(file)
    # image = np.asarray(image)

    # image = cv2.resize(image, (512, 512), cv2.IMREAD_COLOR)
    # # st.image(image)
    # with c1:

    #     st.image(image,caption="image")
    # with c2:
    #     st.image(pred_smoke(model,image),caption="pred")

    if file.name.endswith(('jpg')):    
        image = Image.open(file)
        image = np.asarray(image)
        image = cv2.resize(image, (512, 512), cv2.IMREAD_COLOR)
        with c1:
            st.image(image,caption="image")
        with c2:
            start_time = time.time()
            img = pred_smoke(model,image)
            t = time.time()-start_time
            st.image(img,caption="pred")
            st.write(t)
    if file.name.endswith(('mp4', 'avi')):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        # make_video(tfile.name)
        with c1:
           c
        with c2:
            start_time = time.time()
            make_video(tfile.name)
            t = time.time()-start_time
            video_file = open('pred.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            st.write(n_f/t)
            
