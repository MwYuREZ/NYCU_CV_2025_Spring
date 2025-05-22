# NYCU_CV2025_Spring_HW3
| StudentID |   110550065 |
| --------- | :-----|
| Name  |    尤茂為 |

## Introduction  
In this assignment, we tackle an instance segmentation task on colored medical microscopy images. The goal is to accurately detect and delineate individual cells of four distinct types. 
We build upon the standard `maskrcnn_resnet50_fpn_v2` implementation from TorchVision, but replacement its ResNet50 backbone with a custom Swin-Transformer + FPN encorder. 

## Environment Setup  
Execution environment: Ubuntu 24.04.2  
- Create New Virtual Environment
  ```sh
  python -m venv visual_recog_lab3  
  ```
- Activate the virtual environment
  ```sh
  source visual_recog_lab3/bin/activate
  ```
- Download the required dependencies   
  ```sh
  pip3 install -r requirements.txt  
  ```

## How to exexute  
Clone the repo  
```sh
git clone https://github.com/MwYuREZ/NYCU_CV2025_Spring_HW3.git
```
Download the dataset  
```sh
https://drive.google.com/file/d/1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI/view?usp=sharing
```
Data & Project Structure  
```sh
/NYCU_CV2025_Spring_HW3
├── train_swin.py              # Training script with default anchor settings
├── train_swin_anc.py          # Training script with customized anchors
├── infer_swin.py              # Inference script for default model
├── infer_swin_anc.py          # Inference script for anchor-tuned model
├── /data
│   ├── /test_release          # Test images
│   └── /train                 # Training set (labeled)
│       ├── /<image1_uuid>
│       │   ├── image.tif      # Raw input image
│       │   ├── class1.tif     # Binary mask for cell type 1
│       │   ├── class2.tif     # Binary mask for cell type 2
│       │   ├── class3.tif     # Binary mask for cell type 3
│       │   └── class4.tif     # Binary mask for cell type 4
│       ├── /<image2_uuid>
│       │   └── ...
│       └── ...
```
#### With default anchor setting  
Train:  
```sh
python train_swin.py
```
Test:
```sh
python infer_swin.py
```
#### With customized anchor setting  
Train:  
```sh
python train_swin_anc.py
```
Test:
```sh
python infer_swin_anc.py
```

## Performance Snapshot  
![image](https://github.com/user-attachments/assets/efc31d2c-84b1-4242-85ae-026d9b15c64e)
![image](https://github.com/user-attachments/assets/94651087-7070-42e4-8538-3dbc98f3d22d)  
with private score 0.4307  


