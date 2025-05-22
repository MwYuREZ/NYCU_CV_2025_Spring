#!/usr/bin/env python3
"""
Run Mask R-CNN + Swin-T backbone inference on test_release and generate COCO-format submission.
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tifffile as tiff
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import swin_t, swin_s, swin_b
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from tqdm import tqdm
from pycocotools import mask as maskUtils

# --- CONFIGURATION ---
DATA_ROOT    = 'data'                         # root with test_release/
TEST_MAP     = 'data/test_image_name_to_ids.json'
MODEL_PATH   = 'outputs/swin_best_map50_with_anc.pth'       # load best checkpoint
OUTPUT_JSON  = 'test-results.json'
SCORE_THRESH = 0.5

# --- Swin-T + FPN backbone builder (same as train) ---
def _make_swin_fpn_backbone(trainable_layers=3):
    backbone = swin_s(weights='DEFAULT').features
    for i,layer in enumerate(backbone):
        if i < len(backbone)-trainable_layers:
            for p in layer.parameters(): p.requires_grad_(False)
    return_layers = {'1':'0','3':'1','5':'2','7':'3'}
    in_chs=[96,192,384,768]; out_ch=256
    body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    fpn = FeaturePyramidNetwork(in_channels_list=in_chs, out_channels=out_ch, extra_blocks=LastLevelMaxPool())
    class SwinFPN(nn.Module):
        def __init__(self, body, fpn):
            super().__init__(); self.body=body; self.fpn=fpn; self.out_channels=out_ch
        def forward(self, x):
            feats = self.body(x)
            feats = {k:v.permute(0,3,1,2) for k,v in feats.items()}
            return self.fpn(feats)
    return SwinFPN(body, fpn)

# --- build model (same as train) ---
def get_instance_segmentation_model(num_classes):
    backbone = _make_swin_fpn_backbone(trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=num_classes)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    anchor_sizes = ((4,),(8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 1.5),) * len(anchor_sizes)
    model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return model

# --- inference ---
def run_inference():
    with open(TEST_MAP) as f:
        tests = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_instance_segmentation_model(num_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    results = []
    for entry in tqdm(tests, desc='Inference'):
        fname, img_id = entry['file_name'], entry['id']
        p = Path(DATA_ROOT)/'test_release'/fname
        if not p.exists():
            alt = Path(DATA_ROOT)/'test_release'/str(img_id)/'image.tif'
            p = alt if alt.exists() else None
        if p is None:
            continue

        try:
            arr = tiff.imread(str(p))
        except ValueError:
            arr = np.array(Image.open(str(p)))
        img = Image.fromarray(arr).convert('RGB')
        tensor = T.ToTensor()(img).to(device)
        with torch.no_grad():
            out = model([tensor])[0]

        h, w = img.height, img.width
        for box, lbl, scr, mask in zip(out['boxes'], out['labels'], out['scores'], out['masks']):
            if scr < SCORE_THRESH:
                continue
            x1, y1, x2, y2 = box.cpu().tolist()
            m = (mask[0] > 0.5).cpu().numpy().astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(m))
            rle['counts'] = rle['counts'].decode('utf-8')
            results.append({
                'image_id': int(img_id),
                'category_id': int(lbl.item()),
                'bbox': [x1, y1, x2-x1, y2-y1],
                'score': float(scr.item()),
                'segmentation': {'size': [h, w], 'counts': rle['counts']}
            })

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f)
    print(f"Saved submission to {OUTPUT_JSON}")

if __name__ == '__main__':
    run_inference()
