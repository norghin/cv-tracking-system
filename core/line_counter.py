# core/line_counter.py
import cv2
import numpy as np
import math

class LineType:
    STRAIGHT = "straight"
    L_SHAPED = "l_shaped"

class LineDirection:
    IN = "in"
    OUT = "out"

class LineCounter:
    def __init__(self):
        self.line_type = LineType.STRAIGHT
        self.direction = LineDirection.IN
        
        self.original_points = [(100, 240), (540, 240)]
        
        self.points = list(self.original_points)
        
        self.current_angle = 0
        
        self.counter_in = 0
        self.counter_out = 0
        self.counter_inside = 0
        self.current_inside = 0
        
        self.track_sides = {}
        
        self.line_color = (0, 255, 255)
        self.line_thickness = 2
    
    def set_line(self, line_type, points):
        """Установить новую линию (НЕ сбрасывает угол поворота)."""
        self.line_type = line_type
        self.original_points = [(int(p[0]), int(p[1])) for p in points]
        self._apply_rotation()
    
    def _get_center(self):
        """Вычисляет центр исходной линии."""
        if len(self.original_points) < 2:
            return (0, 0)
        cx = sum(p[0] for p in self.original_points) / len(self.original_points)
        cy = sum(p[1] for p in self.original_points) / len(self.original_points)
        return (cx, cy)
    
    def _apply_rotation(self):
        """Применяет текущий угол к original_points."""
        if len(self.original_points) < 2:
            self.points = list(self.original_points)
            return
        
        cx, cy = self._get_center()
        rad = math.radians(self.current_angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        
        self.points = []
        for x, y in self.original_points:
            dx = x - cx
            dy = y - cy
            nx = cx + dx * cos_a - dy * sin_a
            ny = cy + dx * sin_a + dy * cos_a
            self.points.append((int(nx), int(ny)))
    
    def rotate(self, angle_degrees):
        """Поворачивает линию на указанный абсолютный угол."""
        if len(self.original_points) < 2:
            return
        self.current_angle = int(angle_degrees)
        self._apply_rotation()
    
    def rotate_by(self, delta_degrees):
        """Добавочный поворот."""
        self.rotate(self.current_angle + delta_degrees)
    
    def _cross_product(self, p, a, b):
        """Векторное произведение для определения стороны."""
        return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    
    def _point_side_straight(self, point):
        """Сторона для прямой."""
        return self._cross_product(point, self.points[0], self.points[1])
    
    def _point_side_l_shaped(self, point):
        """Проверка для Г-образной линии."""
        s1 = self._cross_product(point, self.points[0], self.points[1])
        s2 = self._cross_product(point, self.points[1], self.points[2])
        return 1 if (s1 > 0 and s2 > 0) or (s1 < 0 and s2 < 0) else -1
    
    def get_side(self, point):
        """Возвращает 1 или -1 в зависимости от стороны."""
        if self.line_type == LineType.STRAIGHT:
            return 1 if self._point_side_straight(point) > 0 else -1
        else:
            return self._point_side_l_shaped(point)
    
    def update(self, track_id, center_point):
        """Обновляет счётчики для трека."""
        current_side = self.get_side(center_point)
        prev_side = self.track_sides.get(track_id)
        self.track_sides[track_id] = current_side
        
        inside_count = sum(1 for s in self.track_sides.values() if s > 0)
        self.current_inside = inside_count
        
        if prev_side is None:
            return "inside" if current_side > 0 else None
        
        if prev_side < 0 and current_side > 0:
            self.counter_in += 1
            return "in"
        elif prev_side > 0 and current_side < 0:
            self.counter_out += 1
            return "out"
        elif current_side > 0:
            return "inside"
        
        return None
    
    def get_counts(self):
        """Возвращает все счётчики."""
        return {
            "in": self.counter_in,
            "out": self.counter_out,
            "inside": self.current_inside
        }
    
    def reset(self):
        """Сброс счётчиков."""
        self.counter_in = 0
        self.counter_out = 0
        self.current_inside = 0
        self.track_sides.clear()
    
    def draw(self, frame):
        """Рисует линию на кадре."""
        if frame is None or len(self.points) < 2:
            return
        
        for i in range(len(self.points) - 1):
            pt1 = (int(self.points[i][0]), int(self.points[i][1]))
            pt2 = (int(self.points[i+1][0]), int(self.points[i+1][1]))
            cv2.line(frame, pt1, pt2, self.line_color, self.line_thickness)
        
        for point in self.points:
            pt = (int(point[0]), int(point[1]))
            cv2.circle(frame, pt, 5, self.line_color, -1)
            cv2.circle(frame, pt, 7, (0, 0, 0), 1)
        
        cx, cy = self._get_center()
        cv2.circle(frame, (int(cx), int(cy)), 4, (128, 128, 128), -1)