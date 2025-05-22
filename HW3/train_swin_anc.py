#!/usr/bin/env python3
"""
Train Mask R-CNN with a Swin-T + FPN backbone on medical-instance segmentation TIFF data,
and evaluate mAP@50 on a held-out validation split.
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tifffile as tiff
from skimage import measure
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import swin_t, swin_s, swin_b
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from coco_eval import export_coco_gt, evaluate_map50

# --- CONFIGURATION ---
DATA_ROOT  = 'data'    # root containing train/<image_id>/
EPOCHS     = 30        # number of epochs
BATCH_SIZE = 1         # batch size
WORKERS    = 4         # dataloader workers
OUT_DIR    = 'outputs' # checkpoint directory

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# --- Transforms ---
def get_transform(train=False):
    transforms = [T.ToTensor()]
    return T.Compose(transforms)

# --- Dataset ---
class MedicalSegDataset(Dataset):
    def __init__(self, root, image_ids, id_map, transforms=None):
        self.root = Path(root)
        self.ids = image_ids
        self.id_map = id_map
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uuid = self.ids[idx]
        folder = self.root / 'train' / uuid
        # load image
        try:
            arr = tiff.imread(str(folder/'image.tif'))
        except ValueError:
            arr = np.array(Image.open(str(folder/'image.tif')))
        img = Image.fromarray(arr).convert('RGB')

        masks, labels, boxes = [], [], []
        # load each class mask
        for cls in range(1,5):
            mf = folder / f'class{cls}.tif'
            if not mf.exists(): continue
            try:
                m_arr = tiff.imread(str(mf))
            except ValueError:
                m_arr = np.array(Image.open(str(mf)))
            bin_mask = (m_arr>0).astype(np.uint8)
            cc = measure.label(bin_mask)
            for inst in np.unique(cc):
                if inst==0: continue
                m = (cc==inst).astype(np.uint8)
                ys, xs = np.nonzero(m)
                if xs.size==0 or ys.size==0: continue
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                if xmax<=xmin or ymax<=ymin: continue
                masks.append(m)
                labels.append(cls)
                boxes.append([xmin, ymin, xmax, ymax])

        # to tensors
        if masks:
            masks = torch.tensor(np.stack(masks), dtype=torch.uint8)
            labels= torch.tensor(labels,           dtype=torch.int64)
            boxes = torch.tensor(boxes,            dtype=torch.float32)
        else:
            h,w = img.height, img.width
            masks = torch.zeros((0,h,w), dtype=torch.uint8)
            labels= torch.zeros((0,),    dtype=torch.int64)
            boxes = torch.zeros((0,4),    dtype=torch.float32)

        image_id = torch.tensor([self.id_map[uuid]], dtype=torch.int64)
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)

        target = {
            'boxes':      boxes,
            'labels':     labels,
            'masks':      masks,
            'image_id':   image_id,
            'area':       area,
            'iscrowd':    iscrowd
        }

        if self.transforms:
            img = self.transforms(img)
        return img, target


# --- Swin-T + FPN backbone ---
def _make_swin_fpn_backbone(trainable_layers=5):
    backbone = swin_s(weights='DEFAULT').features
    #print(backbone)
    for i,layer in enumerate(backbone):
        if i < len(backbone)-trainable_layers:
            for p in layer.parameters(): p.requires_grad_(False)
    return_layers = {'1':'0','3':'1','5':'2','7':'3'}
    #return_layers = {'0': '0', '1': '1', '2': '2', '3': '3'}
    in_chs = [96,192,384,768]; out_ch=256 #for tiny, small
    #in_chs = [128, 256, 512, 1024]; out_ch=256 #for big
    body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    fpn  = FeaturePyramidNetwork(in_channels_list=in_chs, out_channels=out_ch, extra_blocks=LastLevelMaxPool())
    class SwinFPN(nn.Module):
        def __init__(self, body, fpn):
            super().__init__(); self.body=body; self.fpn=fpn; self.out_channels=out_ch
        def forward(self, x):
            feats = self.body(x)
            feats = {k:v.permute(0,3,1,2) for k,v in feats.items()}
            return self.fpn(feats)
    return SwinFPN(body, fpn)

# --- Model builder ---
def get_instance_segmentation_model(num_classes):
    backbone = _make_swin_fpn_backbone(trainable_layers=5)
    model = MaskRCNN(backbone, num_classes=num_classes)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)

    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 1.5),) * len(anchor_sizes)
    model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    print(f'Total number of parameters: {count_parameters(model)}')
    return model

# --- Training ---
def train_model():
    root = Path(DATA_ROOT)
    all_ids = [d.name for d in (root/'train').iterdir() if d.is_dir()]
    train_ids, val_ids = train_test_split(all_ids, test_size=0.25, random_state=42)
    id_map = {uuid:i+1 for i,uuid in enumerate(train_ids+val_ids)}

    train_ds = MedicalSegDataset(root, train_ids, id_map, get_transform(True))
    val_ds   = MedicalSegDataset(root, val_ids,   id_map, get_transform(False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, collate_fn=lambda x: tuple(zip(*x)))
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=WORKERS, collate_fn=lambda x: tuple(zip(*x)))

    export_coco_gt(val_ids, id_map, DATA_ROOT, 'data/val_instances.json')
    best_map50 = 0.0
    
    train_losses = []
    map50s = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = get_instance_segmentation_model(num_classes=5).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(params, lr=0.9e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    os.makedirs(OUT_DIR, exist_ok=True)
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f'Train {epoch}'):
            imgs = [i.to(device) for i in imgs]
            tgts = [{k:v.to(device) for k,v in t.items()} for t in targets]
            losses = model(imgs, tgts)
            loss = sum(v for v in losses.values())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{EPOCHS} — Train Loss: {total_loss/len(train_loader):.4f}")

        map50 = evaluate_map50(model, val_loader, 'data/val_instances.json', device)
        print(f"Epoch {epoch} — mAP@50: {map50:.3f}")
        
        train_losses.append(total_loss/len(train_loader))
        map50s.append(map50) 
        
        if map50 > best_map50:
            torch.save(model.state_dict(), Path(OUT_DIR)/'swin_best_map50_with_anc.pth')
            best_map50 = map50
            print(f"▶ New best mAP@50: {best_map50:.3f}")
        scheduler.step()
    
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.title('Average Training Loss over Epochs with anchor gen')
    plt.grid(True)
    plt.savefig('Train_Loss_Curve_with_anc.png')

    plt.figure()
    plt.plot(map50s)
    plt.xlabel('Epoch')
    plt.ylabel('Validation mAP@50')
    plt.title('Validation mAP@50 over Epochs with anchor gen')
    plt.grid(True)
    plt.savefig('Validation_mAP50_with_anc.png')

    print('Training curve and Validation mAP with anc saved')

    print("Training complete.")

if __name__ == '__main__':
    train_model()
