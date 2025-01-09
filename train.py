import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kagglehub
from models.resemotenet_enhanced import EnhancedResEmoteNet
from utils.data_utils import RAFDBDataset, get_data_transforms
from utils.train_utils import train_epoch, validate
import shutil
from utils.evaluation_utils import TrainingMonitor

def download_and_prepare_dataset():
    """
    下載並準備 RAF-DB 資料集
    """
    print("正在下載 RAF-DB 資料集...")
    path = kagglehub.dataset_download("shuvoalok/raf-db-dataset")
    print(f"資料集下載到: {path}")
    
    # 找到 DATASET 目錄
    dataset_dir = os.path.join(path, 'DATASET')
    if not os.path.exists(dataset_dir):
        print("找不到 DATASET 目錄，搜尋中...")
        for root, dirs, _ in os.walk(path):
            if 'DATASET' in dirs:
                dataset_dir = os.path.join(root, 'DATASET')
                break
    
    print(f"使用資料集目錄: {dataset_dir}")
    
    # 讀取標籤檔案
    train_labels_file = os.path.join(path, 'train_labels.csv')
    test_labels_file = os.path.join(path, 'test_labels.csv')
    
    labels_dict = {}
    
    # 讀取訓練集標籤
    if os.path.exists(train_labels_file):
        print("讀取訓練集標籤...")
        with open(train_labels_file, 'r') as f:
            next(f)  # 跳過標題行
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1])
                    labels_dict[img_name] = label
    
    # 讀取測試集標籤
    if os.path.exists(test_labels_file):
        print("讀取測試集標籤...")
        with open(test_labels_file, 'r') as f:
            next(f)  # 跳過標題行
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1])
                    labels_dict[img_name] = label
    
    print(f"共讀取 {len(labels_dict)} 個標籤")
    
    # 建立輸出目錄
    output_dir = os.path.join(path, 'processed')
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 處理圖片檔案
    processed_files = 0
    
    # 處理訓練集
    src_train_dir = os.path.join(dataset_dir, 'train')
    if os.path.exists(src_train_dir):
        for label_dir in os.listdir(src_train_dir):
            if label_dir.isdigit():
                src_label_dir = os.path.join(src_train_dir, label_dir)
                dst_label_dir = os.path.join(train_dir, label_dir)
                os.makedirs(dst_label_dir, exist_ok=True)
                
                for img_name in os.listdir(src_label_dir):
                    if img_name.endswith(('.jpg', '.png')):
                        src_path = os.path.join(src_label_dir, img_name)
                        dst_path = os.path.join(dst_label_dir, img_name)
                        shutil.copy2(src_path, dst_path)
                        processed_files += 1
                        
                        if processed_files % 100 == 0:
                            print(f"已處理 {processed_files} 個檔案...")
    
    # 處理測試集
    src_test_dir = os.path.join(dataset_dir, 'test')
    if os.path.exists(src_test_dir):
        for label_dir in os.listdir(src_test_dir):
            if label_dir.isdigit():
                src_label_dir = os.path.join(src_test_dir, label_dir)
                dst_label_dir = os.path.join(test_dir, label_dir)
                os.makedirs(dst_label_dir, exist_ok=True)
                
                for img_name in os.listdir(src_label_dir):
                    if img_name.endswith(('.jpg', '.png')):
                        src_path = os.path.join(src_label_dir, img_name)
                        dst_path = os.path.join(dst_label_dir, img_name)
                        shutil.copy2(src_path, dst_path)
                        processed_files += 1
                        
                        if processed_files % 100 == 0:
                            print(f"已處理 {processed_files} 個檔案...")
    
    # 驗證資料集結構
    train_samples = sum([len(os.listdir(os.path.join(train_dir, d))) 
                        for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))])
    test_samples = sum([len(os.listdir(os.path.join(test_dir, d))) 
                       for d in os.listdir(test_dir) 
                       if os.path.isdir(os.path.join(test_dir, d))])
    
    print(f"\n資料集準備完成:")
    print(f"- 訓練集樣本數: {train_samples}")
    print(f"- 測試集樣本數: {test_samples}")
    print(f"- 總處理檔案數: {processed_files}")
    
    if train_samples == 0 and test_samples == 0:
        raise ValueError("資料集準備失敗：找不到足夠的圖片")
    
    return output_dir

def train(args):
    """
    模型訓練主函式
    """
    # 設定訓練裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用裝置: {device}')
    
    # 設定隨機種子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 初始化訓練監控器
    monitor = TrainingMonitor(
        classes=['生氣', '厭惡', '恐懼', '開心', '傷心', '驚訝', '中性']
    )
    
    try:
        # 下載並準備資料集
        if not args.data_dir:
            args.data_dir = download_and_prepare_dataset()
        
        # 載入訓練資料集
        train_dataset = RAFDBDataset(
            root_dir=args.data_dir,
            transform=get_data_transforms(train=True),
            train=True
        )
        
        print(f"訓練資料集大小: {len(train_dataset)}")
        
        # 載入驗證資料集
        val_dataset = RAFDBDataset(
            root_dir=args.data_dir,
            transform=get_data_transforms(train=False),
            train=False
        )
        
        print(f"驗證資料集大小: {len(val_dataset)}")
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("資料集為空")
        
        # 建立資料入器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 建立模型
        model = EnhancedResEmoteNet(num_classes=7).to(device)
        
        # 定義損失函數和最佳化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 學習率調度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
        
        # 訓練迴圈
        best_acc = 0.0
        for epoch in range(args.epochs):
            print(f'\n第 {epoch+1}/{args.epochs} 個訓練週期')
            
            # 訓練階段
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # 驗證階段
            val_loss, val_acc = validate(
                model, val_loader, criterion, device
            )
            
            # 更新學習率
            scheduler.step()
            
            # 收集預測結果和真實標籤
            all_predictions = []
            all_targets = []
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.numpy())
            
            # 更新監控器
            monitor.update_stats(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=optimizer.param_groups[0]['lr'],
                epoch_predictions=all_predictions,
                epoch_targets=all_targets
            )
            
            # 生成視覺化和報告
            monitor.plot_training_curves(epoch)
            monitor.plot_confusion_matrix(epoch)
            monitor.generate_classification_report(epoch)
            
            # 儲存統計資料
            monitor.save_training_stats()
            
            # 儲存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, 'best_model.pth')
            
            print(f'訓練損失: {train_loss:.4f} 訓練準確率: {train_acc:.4f}')
            print(f'驗證損失: {val_loss:.4f} 驗證準確率: {val_acc:.4f}')
            print(f'最佳驗證準確率: {best_acc:.4f}')
    except Exception as e:
        print(f"訓練過程發生錯誤: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CBAM-FER 訓練程式')
    
    # 資料相關參數
    parser.add_argument('--data_dir', type=str, default='',
                        help='RAF-DB 資料集路徑 (若未指定則自動下載)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='資料載入的行緒數')
    
    # 訓練相關參數
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='訓練週期數')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='起始學習率')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='權重衰減係數')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    
    args = parser.parse_args()
    train(args)