from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
from PIL import Image
import tifffile as tiff
import numpy as np
import json
import os
import torch
from skimage import measure


def export_coco_gt(val_ids, id_map, data_root, out_file):
    """
    val_ids: list of folder names (UUID strings)
    id_map:  dict mapping UUID -> int image_id
    """
    coco = {'images': [], 'annotations': [], 'categories': []}
    ann_id = 1
    for uuid in val_ids:
        img_path = Path(data_root) / 'train' / uuid / 'image.tif'
        w, h    = Image.open(img_path).size
        img_id  = id_map[uuid]
        # *** Use the UUID filename here, not the numeric ID ***
        coco['images'].append({
            'id': img_id,
            'file_name': f"{uuid}.tif",
            'height': h,
            'width': w
        })
        # now annotations
        mask_dir = Path(data_root) / 'train' / uuid
        for cls in range(1,5):
            mfile = mask_dir / f"class{cls}.tif"
            if not mfile.exists(): continue
            arr = tiff.imread(str(mfile))
            bin_mask = (arr>0).astype(np.uint8)
            cc = measure.label(bin_mask)
            for inst in np.unique(cc):
                if inst==0: continue
                m = (cc==inst).astype(np.uint8)
                rle = maskUtils.encode(np.asfortranarray(m))
                rle['counts'] = rle['counts'].decode('utf-8')
                ys,xs = np.nonzero(m)
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cls,
                    'bbox': [float(xmin), float(ymin), float(xmax-xmin), float(ymax-ymin)],
                    'area': float((xmax-xmin)*(ymax-ymin)),
                    'iscrowd': 0,
                    'segmentation': {'size':[h,w], 'counts': rle['counts']}
                })
                ann_id += 1

    for cid in range(1,5):
        coco['categories'].append({'id':cid, 'name':f"class{cid}"})

    with open(out_file,'w') as f:
        json.dump(coco, f)
    print(f"[GT] Wrote COCO GT to {out_file}")


def evaluate_map50(model, val_loader, gt_json, device):
    """
    Runs inference on val_loader, writes a temporary detections JSON,
    and computes mAP@50 (segmentation) against gt_json.
    """
    model.eval()
    preds = []
    for imgs, tgts in val_loader:
        imgs = [img.to(device) for img in imgs]
        with torch.no_grad():
            outs = model(imgs)
        for out, tgt in zip(outs, tgts):
            img_id = int(tgt['image_id'].item())
            h, w   = tgt['masks'].shape[-2:]
            for box, lbl, scr, mask in zip(out['boxes'], out['labels'], out['scores'], out['masks']):
                if scr < 0.05: 
                    continue
                x1,y1,x2,y2 = box.cpu().tolist()
                m = (mask[0]>0.5).cpu().numpy().astype(np.uint8)
                rle = maskUtils.encode(np.asfortranarray(m))
                rle['counts'] = rle['counts'].decode('utf-8')
                preds.append({
                    'image_id': img_id,
                    'category_id': int(lbl.item()),
                    'bbox': [x1,y1,x2-x1,y2-y1],
                    'score': float(scr.item()),
                    'segmentation': {'size':[h,w], 'counts': rle['counts']}
                })

    tmp = 'data/val_preds.json'
    with open(tmp,'w') as f:
        json.dump(preds, f)
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(tmp)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')  # segm for masks
    coco_eval.params.iouThrs = np.array([0.5])             # AP50 only
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return coco_eval.stats[0]