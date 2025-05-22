# NYCU_CV2025_Spring_HW4
| StudentID |   110550065 |
| --------- | :-----|
| Name  |    尤茂為 |

## Introduction  
In this work, we focus on blind restoration of two common weather-induced degradations—rain and snow.
The object is to learn a model that can automatically identify and remove either rain or snow from an input iamge, producing a restored output that closely matches the clean inference. 
Restoration quality is measured by using PSNR, which quantifies the fidelity of the restored iamge relative to its ground truth.  
  
To handle this restoration task, we build upon the recent PromptIR architecture, extending it with a novel HOG-aware attention mechanism. 
This addition leverages histogram-of-oriented-gradients features to better capture local edge and texture patterns, improving the network’s ability to distinguish and remove rain streaks adn snowflakes in a unified framework.

## Environment Setup  
Execution environment: Ubuntu 24.04.2  
- Create New Virtual Environment
  ```sh
  python -m venv visual_recog_lab4  
  ```
- Activate the virtual environment
  ```sh
  source visual_recog_lab4/bin/activate
  ```
- Download the required dependencies   
  ```sh
  pip3 install -r requirements.txt  
  ```

## How to exexute  
Clone the repo  
```sh
git clone https://github.com/MwYuREZ/NYCU_CV_2025_Spring.git
cd HW4  
```
Download the dataset  
```sh
https://drive.google.com/file/d/1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI/view?usp=sharing
```
Data & Project Structure  
```sh
/NYCU_CV2025_Spring_HW4
├── train.py                   # Training script with default 
├── infer.py                   # Inference script  
├── /net
│   ├── model.py               # PromptIR with HOG-aware attention mechanism  
├── /utils  
│   ├── dataset_utils.py       # function of laoding dataset  
│   └── schedulers.py          # LinearWarmupCosineAnnealingLR  
├── /data
│   ├── /test                  # Test images
│   │   ├── /degraded
│   │   │   ├── 0.png
│   │   │   ├── ...
│   │   └───└── 99.png
│   └── /train                 # Training set
│       ├── /clean
│       │   ├── rain_clean-1.png   
│       │   ├── ...   
│       │   ├── rain_clean-1600.png     
│       │   ├── snow_clean-1.png
│       │   ├── ...
│       │   └── snow_clean-1600.png    
│       ├── /degraded
│       │   ├── rain-1.png   
│       │   ├── ...   
│       │   ├── rain-1600.png     
│       │   ├── snow-1.png
│       │   ├── ...
│       └── └── snow-1600.png   
```
#### Execution
Train:  
```sh
python train.py
```
Test:
```sh
python infer.py  
```

## Performance Snapshot  


