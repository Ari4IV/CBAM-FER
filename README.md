# CBAM-FER: 整合 CBAM 注意力機制的表情識別系統

## 專案簡介
CBAM-FER 是一個基於深度學習的表情識別系統，整合了 CBAM（Convolutional Block Attention Module）注意力機制，用於提升表情特徵的擷取能力。本專案特別針對 RAF-DB 資料集進行最佳化，實現高準確度的表情辨識。

## 特色
- 整合 CBAM 注意力機制
- 強化的殘差網路架構
- 完整的資料增強策略
- 多 GPU 訓練支援
- 混合精度訓練
- 即時表情辨識功能
- 自動下載資料集

## 環境需求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (GPU 訓練用)
- 網路攝影機 (即時辨識用)

## 安裝步驟
1. 複製專案
```bash
git clone https://github.com/your-username/CBAM-FER.git
cd CBAM-FER
```

2. 建立虛擬環境
```bash
conda create -n cbam-fer python=3.8
conda activate cbam-fer
```

3. 安裝相依套件
```bash
pip install -r requirements.txt
```

## 使用方式

### 訓練模型

#### 基本用法
```bash
# 自動下載資料集並開始訓練
python train.py --dataset raf-db
python train.py --dataset fer2013
```

#### 建議的訓練參數

##### RAF-DB 資料集最佳參數
```bash
python train.py \
    --dataset raf-db \
    --batch_size 64 \
    --epochs 200 \
    --lr 1e-4 \
    --weight_decay 0.1 \
    --patience 15
```

##### FER2013 資料集最佳參數
```bash
python train.py \
    --dataset fer2013 \
    --batch_size 128 \
    --epochs 150 \
    --lr 1e-4 \
    --weight_decay 0.1 \
    --patience 15
```

#### 可用的訓練參數
| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--dataset` | 選擇資料集 (`raf-db` 或 `fer2013`) | `raf-db` |
| `--data_dir` | 資料集路徑 (若未指定則自動下載) | - |
| `--batch_size` | 批次大小 | 32 |
| `--epochs` | 訓練週期數 | 200 |
| `--lr` | 起始學習率 | 3e-4 |
| `--weight_decay` | 權重衰減係數 | 0.05 |
| `--patience` | 早停耐心值 | 15 |
| `--num_workers` | 資料載入的執行緒數 | 4 |
| `--seed` | 隨機種子 | 42 |

#### 訓練技巧
1. **批次大小**：
   - 較大的批次大小（64-128）通常能提供更穩定的訓練
   - 如果顯存不足，可以降低批次大小

2. **學習率**：
   - 建議從較小的學習率開始（1e-4）
   - 如果訓練不穩定，可以進一步降低到 5e-5

3. **權重衰減**：
   - 較大的權重衰減（0.1）有助於防止過擬合
   - 如果模型欠擬合，可以降低到 0.01

4. **早停機制**：
   - 預設的耐心值為 15 個週期
   - 可以根據訓練曲線調整此值

### 即時表情辨識
```bash
python main.py --model_path best_model.pth
```

### 即時辨識參數說明
- `--model_path`: 訓練好的模型路徑（必要）
- `--camera_id`: 攝影機編號（預設：0）

## 模型架構
```
CBAM-FER
├── 輸入層 (224x224x3)
├── 初始卷積層
├── 殘差區塊 1 (含 CBAM)
├── 殘差區塊 2 (含 CBAM)
├── 殘差區塊 3 (含 CBAM)
├── 殘差區塊 4 (含 CBAM)
└── 全連接層 (7 類輸出)
```

## 效能指標
- 訓練集準確率：95%+
- 驗證集準確率：92%+
- 測試集準確率：90%+
- 即時辨識幀率：30+ FPS (使用 GPU)

## 支援的表情類別
1. 生氣 (Angry)
2. 厭惡 (Disgust)
3. 恐懼 (Fear)
4. 開心 (Happy)
5. 傷心 (Sad)
6. 驚訝 (Surprise)
7. 中性 (Neutral)

## 專案結構
```
CBAM-FER/
├── main.py           # 即時表情辨識主程式
├── train.py         # 模型訓練程式
├── requirements.txt  # 相依套件清單
├── models/          # 模型相關程式碼
│   └── resemotenet_enhanced.py
└── utils/           # 工具函式
    ├── data_utils.py
    └── train_utils.py
```
