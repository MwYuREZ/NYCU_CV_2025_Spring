import torch
import torchvision.models as models
from torch.utils.data import DataLoader
#from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.transforms import ToTensor
import os
from dataset import DigitDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    # Load a pretrained ResNeXt50_32x4d model
    resnext50 = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    
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

def evaluate(model, val_loader, device):
    #model.eval()
    model.train()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    return val_loss / len(val_loader)

def train():
    train_dataset = DigitDataset("nycu-hw2-data/train.json", "nycu-hw2-data/train/", transforms=ToTensor())
    val_dataset = DigitDataset("nycu-hw2-data/valid.json", "nycu-hw2-data/valid/", transforms=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'current device : {device}')
    model = get_model(num_classes=11)  # 10 digits + background
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000105)

    train_losses = []
    val_losses = []
    min_loss = float('inf')

    for epoch in range(9):
        print(f'current epoch : {epoch}')
        model.train()
        train_loss = 0
        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        val_loss = evaluate(model, val_loader, device) 
        avg_train_loss = train_loss / len(train_loader) 
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        ##torch.save(model.state_dict(), f"model/fasterrcnn_digit.pth")
        if val_loss <= min_loss:
            torch.save(model.state_dict(), 'best_model.pth')
            print('best model saved')
            min_loss = val_loss
    
    # Plot training curve
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print('Training curve saved as loss_curve.png')

if __name__ == "__main__":
    train()