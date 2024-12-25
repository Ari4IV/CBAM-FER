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
    
    參數:
        train (bool): 是否為訓練模式
        img_size (int): 圖片尺寸
    """
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
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