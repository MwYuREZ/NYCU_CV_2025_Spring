import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image

from torchvision.models import resnext50_32x4d
import matplotlib.pyplot as plt 

import time




# -----------------------------
# Custom Test Dataset
# -----------------------------
class TestDataset(data.Dataset):
    """
    TestDataset for images in the /data/test directory.
    Note: These images do not have subdirectories as in train/val.
    """
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# -----------------------------
# Main Training & Testing Script
# -----------------------------
def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    start_time = time.time()

    # Hyperparameters
    num_classes = 100
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001492

    transform_train = transforms.Compose([
        transforms.Resize((580, 580)),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomAdjustSharpness(3, p=0.45),
        transforms.RandomResizedCrop(360),
        transforms.ColorJitter(brightness=0.25, contrast=0.22, saturation=0.2, hue=0.15),
        transforms.RandomRotation(35),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((580, 580)),
        transforms.CenterCrop(390),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transform_val

    train_dataset = datasets.ImageFolder(root='data/train', transform=transform_train)
    val_dataset = datasets.ImageFolder(root='data/val', transform=transform_val)
    test_dataset = TestDataset(test_dir='data/test', transform=transform_test)
    
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, Loss, and Optimizer
    # model = ResNet34(num_classes=num_classes).to(device)
    model = resnext50_32x4d(weights='IMAGENET1K_V2').to(device)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False 

    # Unfreeze the last fully connected layer and the last block
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),  # Add dropout
        nn.Linear(512, num_classes)  # Final FC layer for classification
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer = optim.AdamW([
        {'params': model.layer4.parameters(), 'lr': learning_rate * 0.35},
        {'params': model.fc.parameters(), 'lr': learning_rate}
    ], weight_decay=0.01)

    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Decrease learning rate every 5 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

    # Step 2: Count the total number of parameters (no training needed)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Step 3: Print the result
    print(f"Total number of parameters in the model: {num_params}")

    best_val_acc = 0.0

    # for training curve
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # Training and validation loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Collecting all labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        #train_loss = running_loss / total
        train_loss = running_loss
        train_acc = 100. * correct / total


        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        

        # Validation step
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_labels = []
        val_predictions = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

                # Collecting all labels and predictions
                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
        #val_loss = val_loss / total_val
        val_loss = val_loss 
        val_acc = 100. * correct_val / total_val


        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        

        # Modified: Save the metrics for plotting the training curve
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
        scheduler.step()

    # Modified: Plot training curves for loss and accuracy
    epochs_range = range(1, num_epochs+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_history, label='Train Loss')
    plt.plot(epochs_range, val_loss_history, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc_history, label='Train Accuracy')
    plt.plot(epochs_range, val_acc_history, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curve.png")
    #plt.show()
    print("Training curve saved as training_curve.png")
    
    # End time measurement
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training execution time: {total_time:.2f} seconds")

    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Build mapping from index to class name using training dataset info
    # (This replicates the logic: sort classes and assign indices)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    predictions = []
    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            predicted = predicted.cpu().numpy()
            for name, label in zip(img_names, predicted):
                pred_class = idx_to_class[label]
                predictions.append((name, pred_class))

    csv_file = "prediction.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name','pred_label'])
        for name, pred_class in predictions:
            clean_name =  os.path.splitext(name)[0]
            writer.writerow([clean_name, pred_class])
    print(f"Test predictions saved to {csv_file}")

if __name__ == '__main__':
    main()