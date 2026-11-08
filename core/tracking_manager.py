import cv2
import random

def color_for_id(track_id):
    random.seed(track_id)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )

class TrackingManager:
    def __init__(self):
        self.trails = {}

    def process_tracks(self, tracked_objects, frame, params, shared=None, state_lock=None,
                       cam_id=None, active_cam_id=None, fps=30):

        if frame is None or tracked_objects is None:
            return tracked_objects, frame
        
        for obj in tracked_objects:
            # Получаем координаты
            if hasattr(obj, 'tlbr'):
                x1, y1, x2, y2 = [int(v) for v in obj.tlbr]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            elif hasattr(obj, 'tlwh'):
                x, y, w, h = obj.tlwh
                x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
                cx, cy = int(x + w/2), int(y + h/2)
            else:
                continue
            
            local_id = obj.track_id
            gid = getattr(obj, "global_id", None)
            
            # Цвет по локальному ID
            color = color_for_id(local_id)
            
            # Траектория
            if local_id not in self.trails:
                self.trails[local_id] = []
            self.trails[local_id].append((cx, cy))
            if len(self.trails[local_id]) > 50:
                self.trails[local_id].pop(0)
            
            pts = self.trails[local_id]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            
            # Подпись: локальный ID + глобальный ID
            if gid is not None:
                label = f"L:{local_id} G:{gid}"
            else:
                label = f"L:{local_id}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (cx - tw//2 - 5, y1 - th - 10), 
                          (cx + tw//2 + 5, y1 - 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (cx - tw//2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return tracked_objects, frame