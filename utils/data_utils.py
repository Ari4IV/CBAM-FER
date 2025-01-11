import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_data_transforms(train=True, img_size=224):
    """
    取得資料轉換函式
    """
    if train:
        return A.Compose([
            A.RandomResizedCrop(
                size=(img_size, img_size), 
                scale=(0.85, 1.0)  # 縮小隨機裁剪的範圍
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # 降低亮度變化
                contrast_limit=0.1,    # 降低對比度變化
                p=0.5
            ),
            A.Affine(
                scale=(0.95, 1.05),    # 減少縮放範圍
                translate_percent=(-0.05, 0.05),  # 減少平移範圍
                rotate=(-10, 10),       # 減少旋轉角度
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(p=1),  # 簡化 GaussNoise 參數
                A.GaussianBlur(blur_limit=(3, 5), p=1),  # 減少模糊強度
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class RAFDBDataset(Dataset):
    """
    RAF-DB 資料集載入器
    """
    def __init__(self, root_dir, transform=None, train=True):
        """
        參數:
            root_dir (str): 資料集根目錄
            transform: 資料轉換函式
            train (bool): 是否為訓練模式
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # 建立標籤對應
        self.emotion_map = {
            1: 0,  # Angry
            2: 1,  # Disgust
            3: 2,  # Fear
            4: 3,  # Happy
            5: 4,  # Sad
            6: 5,  # Surprise
            7: 6   # Neutral
        }
        
        # 讀取資料集
        self.images = []
        self.labels = []
        
        # 遍歷資料集目錄
        dataset_path = os.path.join(root_dir, 'train' if train else 'test')
        for emotion_id in range(1, 8):  # 7種表情
            emotion_path = os.path.join(dataset_path, str(emotion_id))
            if os.path.exists(emotion_path):
                for img_name in os.listdir(emotion_path):
                    if img_name.endswith(('.jpg', '.png')):
                        self.images.append(os.path.join(emotion_path, img_name))
                        self.labels.append(self.emotion_map[emotion_id])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = self.labels[idx]
        return image, label 

class FER2013Dataset(Dataset):
    """
    FER2013 資料集載入器
    """
    def __init__(self, root_dir, transform=None, train=True):
        """
        參數:
            root_dir (str): 資料集根目錄
            transform: 資料轉換函式
            train (bool): 是否為訓練模式
        """
        self.root_dir = os.path.join(root_dir, 'processed')
        self.transform = transform
        self.train = train
        
        # 建立標籤對應（FER2013 的標籤順序）
        self.emotion_map = {
            0: 0,  # Angry
            1: 1,  # Disgust
            2: 2,  # Fear
            3: 3,  # Happy
            4: 4,  # Sad
            5: 5,  # Surprise
            6: 6   # Neutral
        }
        
        # 讀取資料集
        self.images = []
        self.labels = []
        
        # 遍歷資料集目錄
        dataset_path = os.path.join(self.root_dir, 'train' if train else 'test')
        for emotion_id in range(7):  # 7種表情
            emotion_path = os.path.join(dataset_path, str(emotion_id))
            if os.path.exists(emotion_path):
                for img_name in os.listdir(emotion_path):
                    if img_name.endswith('.png'):
                        self.images.append(os.path.join(emotion_path, img_name))
                        self.labels.append(self.emotion_map[emotion_id])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = self.labels[idx]
        return image, label 