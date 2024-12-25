import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) 注意力機制模組
    結合通道注意力和空間注意力，用於強化特徵圖的重要區域
    
    參數:
        channels (int): 輸入特徵圖的通道數
        reduction_ratio (int): 通道注意力中的降維比例，預設為16
    """
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x (Tensor): 輸入特徵圖 [批次大小, 通道數, 高度, 寬度]
        
        返回:
            Tensor: 經過注意力機制強化後的特徵圖
        """
        x = self.channel_attention(x) * x  # 先進行通道注意力
        x = self.spatial_attention(x) * x  # 再進行空間注意力
        return x

class ChannelAttention(nn.Module):
    """
    通道注意力模組
    使用平均池化和最大池化來獲取通道間的關係
    
    參數:
        channels (int): 輸入特徵圖的通道數
        reduction_ratio (int): 降維比例，用於減少計算量
    """
    def __init__(self, channels, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全域平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全域最大池化
        
        # 多層感知器，用於特徵轉換
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x (Tensor): 輸入特徵圖
            
        返回:
            Tensor: 通道注意力權重
        """
        b, c, _, _ = x.size()
        # 計算平均池化和最大池化的特徵
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out  # 特徵融合
        return torch.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """
    空間注意力模組
    使用通道維度的平均值和最大值來獲取空間注意力圖
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 7x7 卷積核用於空間特徵提取
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x (Tensor): 輸入特徵圖
            
        返回:
            Tensor: 空間注意力權重
        """
        # 計算通道維度的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # 特徵串接
        x = self.conv(x)  # 空間特徵提取
        return torch.sigmoid(x)  # 轉換為注意力權重

class ResidualBlock(nn.Module):
    """
    殘差區塊
    包含兩個卷積層和一個CBAM注意力模組，並具有捷徑連接
    
    參數:
        in_channels (int): 輸入通道數
        out_channels (int): 輸出通道數
        stride (int): 卷積步長，預設為1
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 第一個卷積層
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二個卷積層
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # CBAM注意力模組
        self.cbam = CBAM(out_channels)
        
        # 捷徑連接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x (Tensor): 輸入特徵圖
            
        返回:
            Tensor: 經過殘差區塊處理後的特徵圖
        """
        identity = x
        
        # 主要路徑
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)  # 注意力機制
        
        # 捷徑連接
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

class EnhancedResEmoteNet(nn.Module):
    """
    強化版情緒識別網路
    整合 CBAM 注意力機制與殘差網路的架構
    """
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(EnhancedResEmoteNet, self).__init__()
        
        # 初始卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 殘差區塊
        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        
        # 最佳化：使用較小的特徵維度以減少參數量
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),  # 新增：批次正規化提升穩定性
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 最佳化：權重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """權重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x 