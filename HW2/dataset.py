import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class DigitDataset(Dataset):
    def __init__(self, json_path, img_dir, transforms=None):
        with open(json_path) as f:
            data = json.load(f)
        self.img_dir = img_dir
        self.images = data["images"]
        self.annotations = data["annotations"]
        self.transforms = transforms

        # Build image_id -> annotations mapping
        self.img2anns = {}
        for ann in self.annotations:
            self.img2anns.setdefault(ann["image_id"], []).append(ann)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        img_path = os.path.join(self.img_dir, image_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Build target
        anns = self.img2anns.get(image_id, [])
        boxes = [ann["bbox"] for ann in anns]
        labels = [ann["category_id"] for ann in anns]

        # Convert box format from [x, y, w, h] to [x1, y1, x2, y2]
        boxes = torch.as_tensor([[x, y, x+w, y+h] for x, y, w, h in boxes], dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)
