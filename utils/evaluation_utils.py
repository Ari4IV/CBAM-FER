import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from datetime import datetime
import matplotlib.font_manager as fm
import platform

class TrainingMonitor:
    def __init__(self, output_dir=None, classes=None):
        """
        初始化訓練監控器
        
        Args:
            output_dir: 輸出目錄路徑，如果為 None 則自動創建
            classes: 類別標籤列表
        """
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join('experiments', f'run_{timestamp}')
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.classes = classes or [str(i) for i in range(7)]  # 預設表情類別 0-6
        self.stats = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_predictions': [],  # 儲存每個 epoch 的預測結果
            'epoch_targets': []       # 儲存每個 epoch 的真實標籤
        }
        
        # 設定中文字型
        self._setup_chinese_font()
    
    def _setup_chinese_font(self):
        """設定中文字型"""
        system = platform.system()
        
        if system == 'Windows':
            plt.rcParams['font.family'] = ['Microsoft JhengHei']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = ['Arial Unicode MS']
        elif system == 'Linux':
            plt.rcParams['font.family'] = ['Noto Sans CJK TC']
        
        # 確保 matplotlib 使用 Unicode 支援
        plt.rcParams['axes.unicode_minus'] = False
    
    def update_stats(self, train_loss, train_acc, val_loss, val_acc, lr, 
                    epoch_predictions=None, epoch_targets=None):
        """更新訓練統計資料"""
        self.stats['train_loss'].append(train_loss)
        self.stats['train_acc'].append(train_acc)
        self.stats['val_loss'].append(val_loss)
        self.stats['val_acc'].append(val_acc)
        self.stats['lr'].append(lr)
        
        if epoch_predictions is not None and epoch_targets is not None:
            self.stats['epoch_predictions'].append(epoch_predictions)
            self.stats['epoch_targets'].append(epoch_targets)
    
    def plot_training_curves(self, epoch):
        """繪製訓練曲線"""
        plt.figure(figsize=(15, 5))
        
        # 損失曲線
        plt.subplot(1, 3, 1)
        plt.plot(self.stats['train_loss'], label='訓練損失')
        plt.plot(self.stats['val_loss'], label='驗證損失')
        plt.xlabel('週期')
        plt.ylabel('損失')
        plt.legend()
        plt.title('訓練與驗證損失')
        
        # 準確率曲線
        plt.subplot(1, 3, 2)
        plt.plot(self.stats['train_acc'], label='訓練準確率')
        plt.plot(self.stats['val_acc'], label='驗證準確率')
        plt.xlabel('週期')
        plt.ylabel('準確率')
        plt.legend()
        plt.title('訓練與驗證準確率')
        
        # 學習率曲線
        plt.subplot(1, 3, 3)
        plt.plot(self.stats['lr'])
        plt.xlabel('週期')
        plt.ylabel('學習率')
        plt.title('學習率變化')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'training_curves_epoch_{epoch}.png'))
        plt.close()
    
    def plot_confusion_matrix(self, epoch):
        """繪製混淆矩陣"""
        if not self.stats['epoch_predictions'] or not self.stats['epoch_targets']:
            return
        
        # 獲取最新一輪的預測和標籤
        y_pred = self.stats['epoch_predictions'][-1]
        y_true = self.stats['epoch_targets'][-1]
        
        # 計算混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        
        # 繪製混淆矩陣
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes,
                    yticklabels=self.classes)
        plt.xlabel('預測類別')
        plt.ylabel('真實類別')
        plt.title(f'混淆矩陣 (Epoch {epoch})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()
    
    def generate_classification_report(self, epoch):
        """生成分類報告"""
        if not self.stats['epoch_predictions'] or not self.stats['epoch_targets']:
            return
        
        y_pred = self.stats['epoch_predictions'][-1]
        y_true = self.stats['epoch_targets'][-1]
        
        # 修改：設定 zero_division=1 來避免警告
        report = classification_report(
            y_true, 
            y_pred,
            target_names=self.classes,
            output_dict=True,
            zero_division=1  # 當沒有預測樣本時，設定精確率為 1
        )
        
        # 將報告轉換為 DataFrame 並保存
        df = pd.DataFrame(report).transpose()
        df.to_csv(os.path.join(self.output_dir, f'classification_report_epoch_{epoch}.csv'))
    
    def save_training_stats(self):
        """保存訓練統計資料"""
        # 保存基本統計資料
        basic_stats = {
            'train_loss': self.stats['train_loss'],
            'train_acc': self.stats['train_acc'],
            'val_loss': self.stats['val_loss'],
            'val_acc': self.stats['val_acc'],
            'lr': self.stats['lr']
        }
        df = pd.DataFrame(basic_stats)
        df.to_csv(os.path.join(self.output_dir, 'training_stats.csv'), index=False)
    
    def save_training_config(self, args):
        """保存訓練配置"""
        config = vars(args)
        with open(os.path.join(self.output_dir, 'training_config.json'), 'w', 
                 encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4) 