import cv2
import time
import threading
import numpy as np
import torch
import asyncio
from collections import deque
from queue import Queue, Empty

from web.video_streamer import VideoStreamer
from web.api_server import TrackingAPI

from core.detector import PersonDetector
from core.yolox.byte_tracker import BYTETracker
from core.osnet_extractor import OSNetExtractor
from core.global_reid import GlobalReID
from core.camera_manager import CameraManager, CameraDetector
from core.tracking_manager import TrackingManager
from core.line_counter import LineCounter

from ui.ui_manager import setup_windows, render_main_window, render_metrics, update_controls, handle_keyboard

# -----------------------------
# INIT MODELS
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

detector = PersonDetector(imgsz=320, conf=0.2, device=str(device))
osnet = OSNetExtractor(device=str(device))
reid_model = GlobalReID(
    similarity_threshold=0.55,
    strong_threshold=0.75,
    max_age=30.0,
    max_gallery_size=5,
    alpha=0.3
)

# -----------------------------
# CAMERA STATE
# -----------------------------
class CameraState:
    def __init__(self):
        self.frame = None
        self.display_frame = None
        self.running = True
        self.metrics = CameraMetrics()
        self.tracked_objects = []
        self.reid_queue = Queue(maxsize=20)
        self.line_counter = LineCounter()
        self.line_enabled = False

class CameraMetrics:
    def __init__(self, max_history=100):
        self.fps = deque(maxlen=max_history)
        self.tracks = deque(maxlen=max_history)
        self.avg_age = deque(maxlen=max_history)

class ControlParams:
    def __init__(self):
        self.distance_threshold = 80
        self.hit_counter_max = 60
        self.initialization_delay = 10
        self.min_age_for_count = 12
        self.min_area_factor = 0.015
        self.min_height_factor = 0.25
        self.line_x = 400
        self.margin = 15

# -----------------------------
# MULTI-CAMERA SETUP (ADAPTIVE)
# -----------------------------
# Автоматически определяем количество камер
camera_detector = CameraDetector(max_cameras=10)
NUM_CAMERAS = camera_detector.scan()

if NUM_CAMERAS == 0:
    print("[ERROR] Нет доступных камер! Завершение работы.")
    exit(1)

print(f"[INFO] Запуск системы с {NUM_CAMERAS} камерами")

cam_states = {}
cam_locks = {}
cameras = {}
trackers = {}
tracking_managers = {}
control_params = ControlParams()

for cam_id in camera_detector.get_camera_indices():
    cam_locks[cam_id] = threading.Lock()
    cam_states[cam_id] = CameraState()
    cameras[cam_id] = CameraManager(
        camera_index=cam_id,
        max_retries=100,
        retry_delay=2.0,
        reconnect_timeout=3.0
    )
    tracking_managers[cam_id] = TrackingManager()

# -----------------------------
# WEB INTERFACE
# -----------------------------
video_streamer = VideoStreamer(cam_states, cam_locks, cameras, control_params, reid_model)
api = TrackingAPI(cam_states, cam_locks, cameras, control_params, reid_model, video_streamer)

def run_web_server():
    """Запускает веб-сервер."""
    print("[WEB] Starting web server...")
    api.run(host='0.0.0.0', port=8000)

# -----------------------------
# TRACKER FACTORY
# -----------------------------
def make_tracker():
    args = type('', (), {})()
    args.track_thresh = 0.5
    args.track_buffer = 30
    args.match_thresh = 0.8
    args.mot20 = False
    return BYTETracker(args, frame_rate=30)

# -----------------------------
# ReID WORKER
# -----------------------------
def reid_worker():
    """Фоновый поток для ReID."""
    print("[INFO] ReID worker started")
    
    while True:
        try:
            all_tasks = []
            
            for cam_id in cameras.keys():
                state = cam_states[cam_id]
                if not state.running:
                    continue
                
                while True:
                    try:
                        task = state.reid_queue.get_nowait()
                        all_tasks.append((cam_id, task))
                    except Empty:
                        break
            
            if not all_tasks:
                time.sleep(0.01)
                continue
            
            rois = []
            task_info = []
            
            for cam_id, (obj, roi) in all_tasks:
                if roi is not None and roi.size > 0:
                    rois.append(roi)
                    task_info.append((cam_id, obj))
            
            if rois:
                embeddings = osnet.extract_batch(rois)
                
                for i, (cam_id, obj) in enumerate(task_info):
                    if i < len(embeddings):
                        emb = embeddings[i]
                        if emb is not None:
                            gid = reid_model.match_or_register(emb)
                            obj.global_id = gid
                            
        except Exception as e:
            print(f"[ERROR] ReID worker: {e}")
            time.sleep(0.1)

# -----------------------------
# PROCESSING LOOP
# -----------------------------
def processing_loop(cam_id: int):
    """Основной цикл обработки камеры."""
    camera = cameras[cam_id]
    state = cam_states[cam_id]
    tracking_manager = tracking_managers[cam_id]
    
    # Задержка перед открытием для распределения нагрузки
    time.sleep(cam_id * 0.5)
    
    if not camera.open():
        print(f"[ERROR] Failed to open camera {cam_id}")
        state.running = False
        return
    
    tracker = make_tracker()
    trackers[cam_id] = tracker

    prev_time = time.time()
    frame_count = 0

    print(f"[INFO] Camera {cam_id} started")
    
    while state.running:
        frame = camera.read()
        
        if frame is None:
            if camera.reconnecting:
                with cam_locks[cam_id]:
                    if state.display_frame is not None:
                        msg_frame = state.display_frame.copy()
                        cv2.putText(msg_frame, "RECONNECTING...", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        state.display_frame = msg_frame
            time.sleep(0.01)
            continue
        
        h, w, _ = frame.shape
        frame_count += 1
        
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        display_frame = frame.copy()
        
        try:
            # 1. Детекция
            detections = detector.detect(
                frame,
                control_params.min_area_factor,
                control_params.min_height_factor
            )
            
            output_results = np.array(detections, dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)
            
            # 2. Трекинг
            online_targets = tracker.update(output_results, [h, w], [h, w])
            
            # 3. ReID (каждый 3-й кадр)
            if online_targets and frame_count % 3 == 0:
                for obj in online_targets:
                    if not hasattr(obj, 'is_activated') or not obj.is_activated:
                        continue
                    
                    if hasattr(obj, 'global_id') and obj.global_id is not None:
                        continue
                    
                    if hasattr(obj, 'tlbr'):
                        x1, y1, x2, y2 = [int(v) for v in obj.tlbr]
                    elif hasattr(obj, 'tlwh'):
                        x, y, w_box, h_box = obj.tlwh
                        x1, y1, x2, y2 = int(x), int(y), int(x+w_box), int(y+h_box)
                    else:
                        continue
                    
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    if state.reid_queue.qsize() < 15:
                        try:
                            state.reid_queue.put_nowait((obj, roi))
                        except:
                            pass

            # 4. Отрисовка треков
            if online_targets:
                tracked_objects, display_frame = tracking_manager.process_tracks(
                    tracked_objects=online_targets,
                    frame=display_frame,
                    params=control_params,
                    fps=fps,
                    cam_id=cam_id
                )
                state.tracked_objects = tracked_objects
            
            # 5. Line Counter
            if state.line_enabled:
                if online_targets:
                    for obj in online_targets:
                        if hasattr(obj, 'tlbr'):
                            x1, y1, x2, y2 = [int(v) for v in obj.tlbr]
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        elif hasattr(obj, 'tlwh'):
                            x, y, w_box, h_box = obj.tlwh
                            cx, cy = int(x + w_box/2), int(y + h_box/2)
                        else:
                            continue
                        
                        state.line_counter.update(obj.track_id, (cx, cy))
                
                state.line_counter.draw(display_frame)
                counts = state.line_counter.get_counts()
                y_text = h - 30
                cv2.putText(display_frame, f"IN: {counts['in']}", 
                            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"OUT: {counts['out']}", 
                            (120, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, f"INSIDE: {counts['inside']}", 
                            (240, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 6. ReID info
            reid_active = reid_model.get_active_count()
            cv2.putText(display_frame, f"ReID: {reid_active}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 7. Средний возраст треков
            avg_age = 0.0
            if online_targets:
                ages = [t.age for t in online_targets if hasattr(t, 'age')]
                if ages:
                    avg_age = np.mean(ages)
            
        except Exception as e:
            print(f"[ERROR] Camera {cam_id}: {e}")
            import traceback
            traceback.print_exc()
            online_targets = []
            avg_age = 0.0

        with cam_locks[cam_id]:
            state.frame = frame
            state.display_frame = display_frame
            state.metrics.fps.append(fps)
            state.metrics.tracks.append(len(online_targets))
            state.metrics.avg_age.append(avg_age)

    camera.release()
    print(f"[INFO] Camera {cam_id} stopped")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print(f"[INFO] Starting with {NUM_CAMERAS} cameras + ReID + Web interface...")
    
    setup_windows()
    
    web_thread = threading.Thread(target=run_web_server, daemon=True, name="WebServer")
    web_thread.start()
    time.sleep(0.5)
    print("[INFO] Web dashboard: http://localhost:8000")
    
    reid_thread = threading.Thread(target=reid_worker, daemon=True, name="ReID")
    reid_thread.start()
    
    # Запускаем потоки только для доступных камер
    workers = []
    available_ids = camera_detector.get_camera_indices()
    
    for cam_id in available_ids:
        t = threading.Thread(target=processing_loop, args=(cam_id,), daemon=True, name=f"Cam{cam_id}")
        t.start()
        workers.append(t)
        time.sleep(1.0)  # Увеличенная задержка между запуском камер
    
    time.sleep(2.0) 
    
    active_cam = available_ids[0] if available_ids else 0

    try:
        while True:
            # Проверяем только доступные камеры
            all_running = any(cam_states[cid].running for cid in available_ids)
            if not all_running:
                break
            
            update_controls(control_params)
            
            frames = []
            for cam_id in available_ids:
                with cam_locks[cam_id]:
                    state = cam_states[cam_id]
                    frame = state.display_frame if state.display_frame is not None else state.frame
                    if frame is not None:
                        frames.append(frame.copy())
                    else:
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            if frames:
                h, w, c = frames[0].shape
                grid = np.zeros((h, w * len(frames), c), dtype=np.uint8)
                for i, f in enumerate(frames):
                    grid[:, i*w:(i+1)*w] = f
                
                cv2.putText(grid, f"ACTIVE: Cam {active_cam} | Total cams: {len(available_ids)}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                for i, cam_id in enumerate(available_ids):
                    with cam_locks[cam_id]:
                        state = cam_states[cam_id]
                        if state.metrics.fps:
                            info = f"Cam{cam_id}: {state.metrics.fps[-1]:.0f}FPS | {state.metrics.tracks[-1]}trk"
                            cv2.putText(grid, info, (20 + i*w, h-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                render_main_window(grid)

            with cam_locks[active_cam]:
                state = cam_states[active_cam]
                if state.metrics.fps:
                    metrics_data = {
                        "fps": list(state.metrics.fps),
                        "tracks": list(state.metrics.tracks),
                        "avg_age": list(state.metrics.avg_age),
                    }
                    render_metrics(metrics_data)

            action = handle_keyboard(control_params)
            if action == "exit":
                break
            elif action == "switch":
                # Переключаемся на следующую доступную камеру
                current_idx = available_ids.index(active_cam)
                next_idx = (current_idx + 1) % len(available_ids)
                active_cam = available_ids[next_idx]

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        print("[INFO] Stopping...")
        video_streamer.stop()
        
        for cam_id in available_ids:
            cam_states[cam_id].running = False
        
        for t in workers:
            t.join(timeout=3.0)
        
        cv2.destroyAllWindows()
        print("[INFO] System stopped")