import cv2
import torch
import numpy as np
from models.resemotenet_enhanced import EnhancedResEmoteNet
from utils.data_utils import get_data_transforms
import argparse

class EmotionDetector:
    """
    即時表情辨識器
    """
    def __init__(self, model_path, device='cuda'):
        # 載入模型
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = EnhancedResEmoteNet(num_classes=7).to(self.device)
        self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
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
            
            # 顯示表情和信心度
            text = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                       (0, 255, 0), 2)
        
        # 顯示影像
        cv2.imshow('CBAM-FER 即時表情辨識', frame)
        
        # 按 'q' 結束程式
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()