import cv2
import torch
import numpy as np
from models.resemotenet_enhanced import EnhancedResEmoteNet
from utils.data_utils import get_data_transforms
import argparse
from PIL import Image, ImageDraw, ImageFont

class EmotionDetector:
    """
    即時表情辨識器
    """
    def __init__(self, model_path, device='cuda'):
        # 檢查 CUDA 是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
        # 載入模型
        self.model = EnhancedResEmoteNet(num_classes=7).to(self.device)
        
        # 根據設備載入模型權重
        checkpoint = torch.load(
            model_path, 
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 表情標籤
        self.emotions = {
            0: '生氣', 
            1: '厭惡',
            2: '恐懼',
            3: '開心',
            4: '傷心',
            5: '驚訝',
            6: '中性'
        }
        
        # 資料轉換
        self.transform = get_data_transforms(train=False)
        
        # 載入人臉偵測器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 載入中文字體
        try:
            # 嘗試載入系統字體
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/System/Library/Fonts/STHeiti Light.ttc",  # macOS 備選
                "C:/Windows/Fonts/mingliu.ttc",  # Windows
                "C:/Windows/Fonts/msjh.ttc",  # Windows 備選
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
            ]
            
            self.font = None
            for path in font_paths:
                try:
                    self.font = ImageFont.truetype(path, 32)
                    break
                except:
                    continue
                    
            if self.font is None:
                raise Exception("找不到合適的中文字體")
                
        except Exception as e:
            print(f"警告: 無法載入中文字體: {e}")
            self.font = None

    def draw_text_with_chinese(self, img, text, position, color=(0, 255, 0)):
        if self.font is None:
            # 如果沒有中文字體，退回使用 OpenCV 默認字體
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, color, 2, cv2.LINE_AA)
            return img
            
        # 轉換成 PIL 格式
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 繪製文字
        draw.text(position, text, font=self.font, fill=color[::-1])  # RGB -> BGR
        
        # 轉回 OpenCV 格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def detect_emotion(self, frame):
        """
        偵測單張影像中的表情
        """
        # 轉換成灰階圖片進行人臉偵測
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # 擷取人臉區域
            face = frame[y:y+h, x:x+w]
            
            # 預處理
            augmented = self.transform(image=face)
            face_tensor = augmented['image'].unsqueeze(0).to(self.device)
            
            # 預測
            with torch.no_grad():
                output = self.model(face_tensor)
                prob = torch.softmax(output, dim=1)
                emotion_idx = torch.argmax(prob, dim=1).item()
                confidence = prob[0][emotion_idx].item()
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': self.emotions[emotion_idx],
                'confidence': confidence
            })
            
        return results

def main():
    parser = argparse.ArgumentParser(description='CBAM-FER 即時表情辨識')
    parser.add_argument('--model_path', type=str, required=True,
                        help='訓練好的模型路徑')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='攝影機編號')
    args = parser.parse_args()
    
    # 初始化表情辨識器
    detector = EmotionDetector(args.model_path)
    
    # 開啟攝影機
    cap = cv2.VideoCapture(args.camera_id)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 進行表情辨識
        results = detector.detect_emotion(frame)
        
        # 繪製結果
        for result in results:
            x, y, w, h = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']
            
            # 繪製邊界框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 使用中文字體顯示表情和信心度
            text = f"{emotion} ({confidence:.2f})"
            frame = detector.draw_text_with_chinese(frame, text, (x, y-40))
        
        cv2.imshow('CBAM-FER 即時表情辨識', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()