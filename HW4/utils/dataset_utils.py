import os, re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

def _sorted_numeric(filenames):
    def extract_num(f): 
        m = re.search(r"(\d+)", f)
        return int(m.group(1)) if m else -1
    return sorted(filenames, key=extract_num)

class RainSnowDataset(Dataset):
    def __init__(self, root, transform):
        clean_dir = os.path.join(root, 'train', 'clean')
        deg_dir   = os.path.join(root, 'train', 'degraded')
        # collect rain vs snow by prefix
        self.clean_paths, self.deg_paths, self.prompts = [], [], []
        for typ, idx in [('rain',0), ('snow',1)]:
            cleans = sorted([f for f in os.listdir(clean_dir)   if f.startswith(f"{typ}_clean-")])
            degs   = sorted([f for f in os.listdir(deg_dir)     if f.startswith(f"{typ}-")])
            ##cleans = _sorted_numeric(cleans)
            ##degs   = _sorted_numeric(degs)
            assert len(cleans)==len(degs), f"{typ} count mismatch"
            for c,d in zip(cleans, degs):
                self.clean_paths.append(os.path.join(clean_dir, c))
                self.deg_paths.append(  os.path.join(deg_dir,   d))
                self.prompts.append(idx)
        self.transform = transform

    def __len__(self):
        return len(self.deg_paths)

    def __getitem__(self, i):
        deg = Image.open(self.deg_paths[i]).convert('RGB')
        cln = Image.open(self.clean_paths[i]).convert('RGB')
        return (
            self.transform(deg),
            self.transform(cln),
            torch.LongTensor([self.prompts[i]])
        )
