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

file = st.file_uploader("choose a file", type=['jpg','mp4','avi'])
st.video(file)