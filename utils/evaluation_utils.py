import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

class TrainingMonitor:
    def __init__(self, target_names, dataset_name, log_dir='training_logs'):
        """
        初始化訓練監控器
        
        Args:
            target_names (list): 目標類別的名稱列表
            dataset_name (str): 資料集名稱
            log_dir (str): 日誌檔案的儲存目錄
        """
        self.target_names = target_names
        self.dataset_name = dataset_name
        
        # 建立包含日期時間的日誌目錄
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_dir, f'{dataset_name}_{timestamp}')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化統計資料
        self.stats = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_predictions': [],
            'epoch_targets': []
        }
    
    def update_stats(self, train_loss, train_acc, val_loss, val_acc, lr, epoch_predictions, epoch_targets):
        """更新訓練統計資料"""
        self.stats['train_loss'].append(train_loss)
        self.stats['train_acc'].append(train_acc)
        self.stats['val_loss'].append(val_loss)
        self.stats['val_acc'].append(val_acc)
        self.stats['lr'].append(lr)
        self.stats['epoch_predictions'] = epoch_predictions
        self.stats['epoch_targets'] = epoch_targets
    
    def plot_training_curves(self, epoch):
        """繪製訓練曲線"""
        plt.figure(figsize=(12, 4))
        
        # 損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(self.stats['train_loss'], label='Train Loss')
        plt.plot(self.stats['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 準確率曲線
        plt.subplot(1, 2, 2)
        plt.plot(self.stats['train_acc'], label='Train Acc')
        plt.plot(self.stats['val_acc'], label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'training_curves_epoch_{epoch}.png'))
        plt.close()
    
    def plot_confusion_matrix(self, epoch):
        """繪製混淆矩陣"""
        # 獲取實際出現的類別
        unique_labels = sorted(set(self.stats['epoch_targets']))
        actual_target_names = [self.target_names[i] for i in unique_labels]
        
        cm = confusion_matrix(
            self.stats['epoch_targets'], 
            self.stats['epoch_predictions'],
            labels=unique_labels
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=actual_target_names,
                   yticklabels=actual_target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()
    
    def generate_classification_report(self, epoch):
        """生成分類報告"""
        # 獲取實際出現的類別
        unique_labels = sorted(set(self.stats['epoch_targets']))
        actual_target_names = [self.target_names[i] for i in unique_labels]
        
        report = classification_report(
            self.stats['epoch_targets'],
            self.stats['epoch_predictions'],
            labels=unique_labels,  # 指定實際的標籤
            target_names=actual_target_names,  # 使用對應的名稱
            digits=4,
            zero_division=0  # 明確指定除零時的行為
        )
        
        # 更新報告格式，加入資料集資訊
        with open(os.path.join(self.log_dir, f'classification_report_epoch_{epoch}.txt'), 'w') as f:
            f.write(f"資料集：{self.dataset_name}\n")
            f.write(f"日期時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分類報告 - 第 {epoch} 個訓練週期\n")
            f.write("="*50 + "\n")
            f.write("註：如果某些類別沒有預測樣本，精確度將被設為 0\n")
            f.write("="*50 + "\n\n")
            f.write(report)
    
    def save_training_stats(self):
        """儲存訓練統計資料"""
        stats_file = os.path.join(self.log_dir, 'training_stats.json')
        save_stats = {
            'dataset_name': self.dataset_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': self.stats['train_loss'],
            'train_acc': self.stats['train_acc'],
            'val_loss': self.stats['val_loss'],
            'val_acc': self.stats['val_acc'],
            'lr': self.stats['lr']
        }
        
        with open(stats_file, 'w') as f:
            json.dump(save_stats, f) 