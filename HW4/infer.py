import os, argparse, numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils.dataset_utils import _sorted_numeric

from net.model import PromptIR

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)  # points at data
    p.add_argument('--ckpt',      default='best_allinone_IR.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model  = PromptIR(decoder=True).to(device)
    #model.load_state_dict(torch.load(args.ckpt, map_location=device))
    from train import RainSnowModule   # wherever your LightningModule lives
    
    # load the LightningModule (it will restore optimizer, etc. but you only need .net)
    lit_model = RainSnowModule.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        map_location=device
    )
    model = lit_model.net.to(device)
    model.eval()

    transform = transforms.ToTensor()
    test_dir = os.path.join(args.data_root, 'test', 'degraded')
    files    = _sorted_numeric(os.listdir(test_dir))
    out_dict = {}

    with torch.no_grad():
        for fname in files:
            img = Image.open(os.path.join(test_dir, fname)).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            pidx = torch.LongTensor([0]).to(device)

            pred = model(inp, pidx).clamp(0,1).cpu().squeeze(0).numpy()
            pred = (pred * 255).round().astype(np.uint8)
            out_dict[fname] = pred

    np.savez('pred.npz', **out_dict)
    print(f"Saved {len(out_dict)} images to pred.npz")
    
    #####output image
    out_dir = os.path.join(args.data_root, 'res')
    os.makedirs(out_dir, exist_ok=True)

    keys = list(out_dict.keys())
    n = len(keys)
    ##first5 and last5
    sample_keys = keys[:5] + keys[max(5, n-5):n]

    for fname in sample_keys:
        img_pred = out_dict[fname]
        if img_pred.ndim == 3:
          img_to_save = img_pred.transpose(1, 2, 0) 
        else:
          img_to_save = img_pred
        save_path = os.path.join(out_dir, fname)
        Image.fromarray(img_to_save).save(save_path)

    print(f"Saved {len(sample_keys)} sample images (first 5 + last 5) to: {out_dir}")

if __name__=='__main__':
    main()
