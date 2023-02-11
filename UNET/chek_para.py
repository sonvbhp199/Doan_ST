from prettytable import PrettyTable
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
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
    print(table)
    return f"{total_params:,}"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "files/checkpoint.pth"
model = build_unet()
model = model.to(device)
checkpoint = torch.load(checkpoint_path,map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
count_parameters(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)