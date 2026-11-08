import cv2
import torch
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model=None, model_path="yolov8n.pt", imgsz=320, conf=0.2, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[INFO] PersonDetector using device: {self.device}")
        
        if model is None:
            self.model = YOLO(model_path)
        else:
            self.model = model
            
        self.imgsz = imgsz
        self.conf = conf

        self._warmup()

    def _warmup(self):
        """Прогрев модели для более быстрой первой детекции."""
        print("[INFO] Warming up detector...")
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[INFO] Detector ready")

    def detect(self, frame, min_area_factor=0.01, min_height_factor=0.1):
        """
        Детекция людей с оптимизацией скорости.
        """
        if frame is None:
            return []
        
        h, w, _ = frame.shape
        
        scale = 1.0
        if w > 640:
            scale = 640.0 / w
            new_w = 640
            new_h = int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))
        else:
            frame_resized = frame
            new_w, new_h = w, h

        results = self.model(
            frame_resized, 
            imgsz=self.imgsz, 
            conf=self.conf, 
            verbose=False,
            device=self.device,
            half=True if self.device.type == 'cuda' else False
        )[0]

        detections = []
        min_area = new_w * new_h * min_area_factor
        min_height = new_h * min_height_factor
        
        if results.boxes is not None:
            # Быстрый доступ к данным
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                if int(cls) != 0:  # Только люди
                    continue
                    
                x1, y1, x2, y2 = box
                
                box_w = x2 - x1
                box_h = y2 - y1
                
                if box_w * box_h < min_area or box_h < min_height:
                    continue
                
                if scale != 1.0:
                    x1, x2 = x1/scale, x2/scale
                    y1, y2 = y1/scale, y2/scale
                
                # Клиппинг
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2, float(score), 1])

        return detections