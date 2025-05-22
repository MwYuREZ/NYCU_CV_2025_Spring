# NYCU_CV2025_Spring_HW2  
| StudentID |   110550065 |
| --------- | :-----|
| Name  |    尤茂為 |

## Introduction  
In this task, we aim to perform digit recognition on a dataset consisting RGB images where each image contains multiple digits. The objective is twofold : (1) Detect each digit in the image by outputting bounding boxes and class labels. (2) Use the results from Task 1 (pred.json) as input to predict the whole number. My approach leverages the state-of-the-art object detection framework Faster R-CNN, which contains ResNeXt50_32x4d as backbone.  

## How to Install  
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install matplotlib  
```

## How to exexute 
Train:  
```sh
python train.py
```
Test:
```sh
python infer.py
```

## Performance Snapshot  
![image](https://github.com/user-attachments/assets/564ad074-c0b4-446f-ae23-363ed8ad394b)

