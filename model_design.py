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
config_vit.transformer.num_layers = 3
config_vit.n_classes = num_classes
config_vit.n_skip = 0
if vit_name.find("R50") != -1:
    config_vit.n_skip = n_skip
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
        
for name, param in net.named_parameters():
    if "encoder" in name:
        param.requires_grad = False

for name, param in net.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name} | Shape: {param.shape}")

if os.path.exists(config_vit.pretrained_path):
    print("pretrained_path", config_vit.pretrained_path)
    net.load_from(weights=np.load(config_vit.pretrained_path))

elif config_vit.get("pretrained_path_alt1") and os.path.exists(config_vit.get("pretrained_path_alt1")):
    print("pretrained_path", config_vit.pretrained_path_alt1)
    
    weights = np.load(config_vit.pretrained_path_alt1)
    print("weights", type(weights))
    
    for name, param in weights.items():
        print(f"Layer: {name} | Shape: {param.shape}")
        
    net.load_from(weights=weights)
    
    
import logging
import collections
from datasets import UAV_HSI_Crop_dataset, RandomGenerator
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

        
logging.basicConfig(filename='output.log', level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s')

root_path = r"D:\OneDrive\The University of Memphis\Aaron L Robinson (alrobins) - Mazhar_1\muas_eece\Datasets\UAV-HSI-Crop-Dataset"
db_train = UAV_HSI_Crop_dataset(
    base_dir=root_path,
    transform=transforms.Compose([RandomGenerator([img_size, img_size])]),
)


def worker_init_fn(worker_id):
    random.seed(seed + worker_id)

num_workers = 4
batch_size = 16
train_loader = DataLoader(
    db_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    # worker_init_fn=worker_init_fn,
)

lr_ = base_lr = 0.01

net.train()
optimizer = optim.SGD(
    net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001
) # weight_decay=0.0001
scaler = GradScaler()
ce_loss = CrossEntropyLoss(ignore_index=4)


max_epochs, my_gradients, my_loss = 10, collections.defaultdict(list), []
logging.info("="*80)
logging.info("Start new experiment.")
logging.info("="*80)

for epoch_num in range(1, max_epochs + 1):
    for i_batch, sampled_batch in enumerate(train_loader):
        
        volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        volume_batch, label_batch = volume_batch.to(dev), label_batch.to(dev)

        optimizer.zero_grad()
        with autocast():
            outputs = net(volume_batch)
    
            loss = ce_loss(outputs, label_batch[:].long())
            logging.info("="*60)
            logging.info(f"Epoch {epoch_num:04d} batch {i_batch:03d}: loss {loss.item()}")
            my_loss.append(loss.item())
        
        scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        for name, param in net.named_parameters():
            if "weight" in name and param.grad is not None:
                val = param.grad.abs().mean()
                logging.info(f"Layer: {name} | gradient: {val}")
                my_gradients[name].append(val.item())
                
        scaler.step(optimizer)
        scaler.update()
        
    logging.info("="*80)
    logging.info(f"Epoch {epoch_num:04d}: loss {np.array(my_loss).mean()}")
    logging.info("="*80)
        
for k, v in my_gradients.items():
    # v = [v1.item() for v1 in v]
    logging.info("Layer: %s | gradient: %s %s", k, np.array(v).mean(), v)
        