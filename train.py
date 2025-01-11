import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kagglehub
from models.resemotenet_enhanced import EnhancedResEmoteNet
from utils.data_utils import RAFDBDataset, get_data_transforms
import shutil
from utils.evaluation_utils import TrainingMonitor
from tqdm import tqdm
from utils.train_utils import LabelSmoothingLoss, AverageMeter

def download_and_prepare_dataset(dataset_name):
    """
    下載並準備資料集
    """
    if dataset_name == 'raf-db':
        return download_and_prepare_rafdb()
    elif dataset_name == 'fer2013':
        return download_and_prepare_fer2013()
    else:
        raise ValueError(f"不支援的資料集: {dataset_name}")

def download_and_prepare_fer2013():
    """
    下載並準備 FER-2013 資料集
    """
    print("正在下載 FER-2013 資料集...")
    path = kagglehub.dataset_download("msambare/fer2013")
    
    # 建立輸出目錄
    output_dir = os.path.join(path, 'processed')
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 直接使用原始目錄結構，不轉換標籤
    for src_dir, dst_dir in [(os.path.join(path, 'train'), train_dir), 
                            (os.path.join(path, 'test'), test_dir)]:
        if os.path.exists(src_dir):
            for emotion_dir in os.listdir(src_dir):
                src_emotion_dir = os.path.join(src_dir, emotion_dir)
                dst_emotion_dir = os.path.join(dst_dir, emotion_dir)  # 保持原始目錄名
                os.makedirs(dst_emotion_dir, exist_ok=True)
                
                for img_name in os.listdir(src_emotion_dir):
                    if img_name.endswith(('.jpg', '.png')):
                        src_path = os.path.join(src_emotion_dir, img_name)
                        dst_path = os.path.join(dst_emotion_dir, img_name)
                        shutil.copy2(src_path, dst_path)
    
    return output_dir

def download_and_prepare_rafdb():
    """
    下載並準備 RAF-DB 資料集 (原本的函數內容)
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

    print(f"\n總共處理了 {processed_files} 個檔案")
    
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
    
    # 根據資料集設定標籤名稱和監控器
    if args.dataset == 'fer2013':
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        monitor = TrainingMonitor(
            target_names=emotion_labels,
            dataset_name=args.dataset
        )
    else:  # raf-db
        monitor = TrainingMonitor(
            target_names=['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral'],
            dataset_name=args.dataset
        )
    
    try:
        # 根據指定的資料集下載和準備資料
        data_dir = args.data_dir if args.data_dir else download_and_prepare_dataset(args.dataset)
        
        # 載入訓練資料集
        train_dataset = RAFDBDataset(
            root_dir=data_dir,
            transform=get_data_transforms(train=True),
            train=True
        )
        
        print(f"訓練資料集大小: {len(train_dataset)}")
        
        # 載入驗證資料集
        val_dataset = RAFDBDataset(
            root_dir=data_dir,
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
        criterion = LabelSmoothingLoss(classes=7, smoothing=0.1)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 使用 OneCycleLR 替代 CosineAnnealingWarmRestarts
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        # 初始化早停相關變數
        best_val_acc = 0
        patience_counter = 0
        
        # 訓練迴圈
        for epoch in range(args.epochs):
            print(f'\n第 {epoch+1}/{args.epochs} 個訓練週期')
            
            # 訓練階段
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # 使用 tqdm 來顯示進度條
            train_loop = tqdm(train_loader, desc=f"訓練 Epoch {epoch+1}/{args.epochs}")
            for data in train_loop:
                inputs, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新進度條資訊
                train_loop.set_postfix(loss=loss.item(), acc=correct/total)
            
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            
            # 驗證階段
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"驗證 Epoch {epoch+1}/{args.epochs}")
                for data in val_loop:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_loop.set_postfix(loss=loss.item(), acc=val_correct/val_total)
            
            val_loss = val_running_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # 早停檢查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 儲存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_val_acc,
                }, 'best_model.pth')
                print(f"儲存新的最佳模型，驗證準確率: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"驗證準確率未改善，已經 {patience_counter} 個週期")
            
            if patience_counter >= args.patience:
                print(f"\n早停：驗證準確率連續 {args.patience} 個週期未改善")
                break
            
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
            
            print(f'訓練損失: {train_loss:.4f} 訓練準確率: {train_acc:.4f}')
            print(f'驗證損失: {val_loss:.4f} 驗證準確率: {val_acc:.4f}')
            print(f'最佳驗證準確率: {best_val_acc:.4f}')
    except Exception as e:
        print(f"訓練過程發生錯誤: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='表情識別訓練程式')
    
    # 資料相關參數
    parser.add_argument('--dataset', type=str, default='raf-db', choices=['raf-db', 'fer2013'],
                        help='選擇資料集 (raf-db 或 fer2013)')
    parser.add_argument('--data_dir', type=str, default='',
                        help='資料集路徑 (若未指定則自動下載)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='資料載入的行緒數')
    
    # 訓練相關參數
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='訓練週期數')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='起始學習率')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='權重衰減係數')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    
    args = parser.parse_args()
    train(args)