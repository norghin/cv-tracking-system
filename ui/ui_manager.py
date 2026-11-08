import cv2
import numpy as np

def nothing(x):
    pass

def setup_windows():
    """Создает окна с элементами управления."""
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 500)
    
    cv2.namedWindow("Metrics", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Metrics", 600, 300)
    
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 1280, 520)

    cv2.createTrackbar("distance_threshold", "Controls", 80, 200, nothing)
    cv2.createTrackbar("hit_counter_max", "Controls", 60, 200, nothing)
    cv2.createTrackbar("initialization_delay", "Controls", 10, 50, nothing)
    cv2.createTrackbar("min_age", "Controls", 12, 100, nothing)

    cv2.createTrackbar("min_area_x1000", "Controls", 15, 100, nothing)
    cv2.createTrackbar("min_height_percent", "Controls", 25, 100, nothing)

    cv2.createTrackbar("line_x", "Controls", 400, 1920, nothing)
    cv2.createTrackbar("margin", "Controls", 15, 200, nothing)


def draw_graph(metrics_img, values, y_offset, color, label):
    """
    Рисует график метрик с защитой от ошибок.
    """
    if len(values) > 200:
        values = values[-200:]
    
    if len(values) < 2:
        return

    max_val = max(values) if values else 1
    if max_val <= 0 or np.isnan(max_val) or np.isinf(max_val):
        max_val = 1

    cv2.rectangle(metrics_img, (0, y_offset - 100), (600, y_offset), (30, 30, 30), -1)
    
    cv2.line(metrics_img, (0, y_offset), (600, y_offset), (50, 50, 50), 1)

    for i in range(1, len(values)):
        val_prev = values[i - 1]
        val_curr = values[i]
        
        if np.isnan(val_prev) or np.isinf(val_prev):
            val_prev = 0
        if np.isnan(val_curr) or np.isinf(val_curr):
            val_curr = 0
        
        x1 = int((i - 1) / (len(values) - 1) * 600)
        y1 = int(y_offset - (val_prev / max_val) * 80)
        
        x2 = int(i / (len(values) - 1) * 600)
        y2 = int(y_offset - (val_curr / max_val) * 80)
        
        y1 = max(y_offset - 100, min(y_offset, y1))
        y2 = max(y_offset - 100, min(y_offset, y2))
        
        cv2.line(metrics_img, (x1, y1), (x2, y2), color, 2)

    current_val = values[-1] if values else 0
    if not np.isnan(current_val) and not np.isinf(current_val):
        cv2.putText(
            metrics_img,
            f"{label}: {current_val:.2f}",
            (10, y_offset - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    cv2.putText(
        metrics_img,
        f"Max: {max_val:.2f}",
        (10, y_offset - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1
    )


def update_controls(params):
    """
    Обновляет параметры из трекбаров.
    
    Args:
        params: объект ControlParams
    """
    if params is None:
        return
        
    params.distance_threshold = cv2.getTrackbarPos("distance_threshold", "Controls")
    params.hit_counter_max = cv2.getTrackbarPos("hit_counter_max", "Controls")
    params.initialization_delay = cv2.getTrackbarPos("initialization_delay", "Controls")
    params.min_age_for_count = cv2.getTrackbarPos("min_age", "Controls")

    params.min_area_factor = cv2.getTrackbarPos("min_area_x1000", "Controls") / 1000.0
    params.min_height_factor = cv2.getTrackbarPos("min_height_percent", "Controls") / 100.0

    params.line_x = cv2.getTrackbarPos("line_x", "Controls")
    params.margin = cv2.getTrackbarPos("margin", "Controls")


def render_main_window(frame):
    """Отображает основное окно с трекингом."""
    if frame is not None and frame.size > 0:
        cv2.imshow("Tracking", frame)


def render_metrics(metrics_data):
    """
    Отображает окно с графиками метрик.
    """
    metrics_img = np.zeros((300, 600, 3), dtype=np.uint8)
    
    for key in metrics_data:
        metrics_data[key] = [
            v for v in metrics_data[key] 
            if not np.isnan(v) and not np.isinf(v)
        ]

    if metrics_data.get("fps"):
        draw_graph(metrics_img, metrics_data["fps"], 100, (0, 255, 0), "FPS")
    
    if metrics_data.get("tracks"):
        draw_graph(metrics_img, metrics_data["tracks"], 200, (255, 0, 0), "Tracks")
    
    if metrics_data.get("avg_age"):
        draw_graph(metrics_img, metrics_data["avg_age"], 300, (0, 0, 255), "Avg Age")

    cv2.imshow("Metrics", metrics_img)


def handle_keyboard(params):
    """Обрабатывает нажатия клавиш."""
    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord('q'):
        return "exit"
    
    if key == ord(' '):
        return "switch"
    
    if key == ord('1'):
        return "cam_0"
    
    if key == ord('2'):
        return "cam_1"
    
    if key == ord('r'):
        return "reconnect"
    
    if key == ord('R'):
        return "reconnect_0"

    return None