import torch
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.transforms import ToTensor
from PIL import Image
import os, json, pandas as pd
from tqdm import tqdm

def get_model(num_classes):
    # Load a pretrained ResNeXt50_32x4d model
    resnext50 = models.resnext50_32x4d(pretrained=True)
    
    # Create a backbone with FPN using layers from ResNeXt50.
    # The return_layers mapping specifies which layers to use.
    backbone = BackboneWithFPN(
        resnext50,
        return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
        in_channels_list=[256, 512, 1024, 2048],  # Channels for each layer in ResNeXt50
        out_channels=256  # This is the number of channels that FPN will output
    )
    
    # Create Faster R-CNN using the custom backbone.
    model = FasterRCNN(backbone, num_classes=num_classes)
    
    # Replace the box predictor with FastRCNNPredictor (if needed) to match the number of classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def load_model(path, num_classes):
    #model = fasterrcnn_resnet50_fpn(pretrained=False)
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = get_model(num_classes)
    model.load_state_dict(torch.load(path))
    #model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    return model

def main():
    test_dir = "nycu-hw2-data/test/"
    model = load_model("best_model.pth", 11)
    transform = ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_files = sorted(os.listdir(test_dir))
    pred_json = []
    pred_csv = []

    #image_id = 1
    for file_name in tqdm(image_files):
        image_id = int(os.path.splitext(file_name)[0])
        img = Image.open(os.path.join(test_dir, file_name)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        digits = []
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.5:
                continue
            x_min, y_min, x_max, y_max = box
            pred_json.append({
                "image_id": image_id,
                "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                "score": float(score),
                "category_id": int(label)
            })
            digits.append((x_min, int(label)))

        if digits:
            digits.sort()
            pred_label = int("".join(str(d[1] - 1) for d in digits))  # category_id - 1 = digit
        else:
            pred_label = -1

        pred_csv.append({
            "image_id": image_id,
            "pred_label": pred_label
        })

        ##image_id += 1

    with open("pred.json", "w") as f:
        json.dump(pred_json, f)

    pd.DataFrame(pred_csv).to_csv("pred.csv", index=False)

if __name__ == "__main__":
    main()
