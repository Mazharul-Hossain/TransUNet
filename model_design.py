# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file to design network.

Created on Thu Jan 23 17:46:03 2025
@author: mazhar
"""
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


cudnn.benchmark = False
cudnn.deterministic = True
seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

vit_name = "R50+ViT-B_16"
num_classes = 30
n_skip = 3
img_size = 96 
vit_patches_size = 16

config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = num_classes
config_vit.n_skip = n_skip
if vit_name.find("R50") != -1:
    config_vit.patches.grid = (
        int(img_size / vit_patches_size),
        int(img_size / vit_patches_size),
    )
print("config_vit", type(config_vit), config_vit)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("dev", dev)

net = ViT_seg(
        config_vit, img_size=img_size, num_classes=config_vit.n_classes
).to(dev)
print("net", net)

for name, param in net.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name} | Shape: {param.shape}") 
        #  | Values: {param.data}

if os.path.exists(config_vit.pretrained_path):
    print("pretrained_path", config_vit.pretrained_path)
    net.load_from(weights=np.load(config_vit.pretrained_path))

elif os.path.exists(config_vit.pretrained_path_alt1):
    print("pretrained_path", config_vit.pretrained_path_alt1)
    
    weights = np.load(config_vit.pretrained_path_alt1)
    print("weights", type(weights))
    
    for name, param in weights.items():
        print(f"Layer: {name} | Shape: {param.shape}")
        
    net.load_from(weights=weights)
