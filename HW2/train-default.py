import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import ToTensor
import os
from dataset import DigitDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    # Load the model with pretrained weights
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for our custom number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def evaluate(model, val_loader, device):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    min_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(6):
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
        
        lr_scheduler.step()

        val_loss = evaluate(model, val_loader, device) 
        avg_train_loss = train_loss / len(train_loader) 
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if val_loss <= min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), 'best_default_model.pth')
            print('best model saved')

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
