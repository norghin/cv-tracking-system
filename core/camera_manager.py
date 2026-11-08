import cv2
import time
import numpy as np
import platform


class CameraManager:
    # Словарь на уровне класса для отслеживания занятых ИНДЕКСОВ
    _used_indices = set()
    
    def __init__(
        self,
        camera_index=0,
        fallback_index=None,
        width=640,
        height=480,
        backend=None,
        max_retries=10,
        retry_delay=1.0,
        warmup_frames=5,
        reconnect_timeout=3.0
    ):
        self.camera_index = camera_index
        self.original_index = camera_index
        self.fallback_index = fallback_index
        self.width = width
        self.height = height
        self.backend = backend
        self.cap = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.warmup_frames = warmup_frames
        self.reconnect_timeout = reconnect_timeout
        self.is_opened = False
        self.last_frame_time = 0
        self.reconnecting = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 30
        self.actual_index = None
        
    def _get_backends(self):
        backends = []
        
        if self.backend is not None:
            backends.append(self.backend)
        
        system = platform.system()
        if system == "Windows":
            backends.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY])
        else:
            backends.extend([cv2.CAP_V4L2, cv2.CAP_ANY])
        
        return backends

    def _list_all_camera_indices(self, max_check=10):
        """Возвращает ВСЕ доступные индексы камер (даже занятые)"""
        available = []
        for idx in range(max_check):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Проверяем, что камера реально отдаёт кадры
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        available.append(idx)
                        break
                    time.sleep(0.05)
            cap.release()
        return available

    def _try_open_index(self, idx):
        """Пробует открыть камеру по конкретному индексу"""
        for backend in self._get_backends():
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    # Проверяем, что камера реально работает
                    for _ in range(5):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            # Устанавливаем разрешение
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            return cap
                        time.sleep(0.05)
                    cap.release()
            except:
                try:
                    cap.release()
                except:
                    pass
                continue
        return None

    def open(self) -> bool:
        if self.is_opened and self.cap is not None:
            return True
        if self.cap is not None:
            self.release()
        
        # Стратегия 1: пробуем исходный индекс
        if self.camera_index not in CameraManager._used_indices:
            print(f"[INFO] Пробуем индекс {self.camera_index}...")
            cap = self._try_open_index(self.camera_index)
            if cap is not None:
                self.cap = cap
                self.actual_index = self.camera_index
                CameraManager._used_indices.add(self.camera_index)
                print(f"[INFO] Камера открыта на индексе {self.camera_index}")
                self._finalize_open()
                return True
        
        # Стратегия 2: пробуем fallback
        if self.fallback_index is not None and self.fallback_index not in CameraManager._used_indices:
            print(f"[INFO] Пробуем fallback индекс {self.fallback_index}...")
            cap = self._try_open_index(self.fallback_index)
            if cap is not None:
                self.cap = cap
                self.actual_index = self.fallback_index
                self.camera_index = self.fallback_index
                CameraManager._used_indices.add(self.fallback_index)
                print(f"[INFO] Fallback камера открыта на индексе {self.fallback_index}")
                self._finalize_open()
                return True
        
        # Стратегия 3: ищем ЛЮБОЙ свободный индекс
        print(f"[INFO] Сканируем свободные индексы (заняты: {CameraManager._used_indices})...")
        all_indices = self._list_all_camera_indices(max_check=10)
        print(f"[INFO] Все доступные индексы в системе: {all_indices}")
        
        for idx in all_indices:
            if idx not in CameraManager._used_indices:
                cap = self._try_open_index(idx)
                if cap is not None:
                    self.cap = cap
                    self.actual_index = idx
                    self.camera_index = idx
                    CameraManager._used_indices.add(idx)
                    print(f"[INFO] Камера открыта на индексе {idx} (исходный был {self.original_index})")
                    self._finalize_open()
                    return True
        
        print(f"[ERROR] Все индексы заняты или камеры недоступны. Занято: {CameraManager._used_indices}")
        return False
    
    def _finalize_open(self):
        """Завершающие операции после открытия камеры"""
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Камера {self.actual_index}: {actual_w}x{actual_h}")
        
        # Прогрев камеры
        for i in range(self.warmup_frames):
            self.cap.read()
            time.sleep(0.03)
        
        self.is_opened = True
        self.last_frame_time = time.time()
        self.consecutive_errors = 0
        self.reconnecting = False
    
    def _backend_name(self, backend):
        names = {
            cv2.CAP_ANY: "ANY",
            cv2.CAP_DSHOW: "DSHOW",
            cv2.CAP_MSMF: "MSMF",
            cv2.CAP_V4L2: "V4L2",
        }
        return names.get(backend, f"Unknown({backend})")

    def read(self):
        if not self.is_opened or self.cap is None:
            # Пробуем переподключиться на лету
            if self.reconnect():
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    return frame
            return None
        
        try:
            ret, frame = self.cap.read()

            if ret and frame is not None and frame.size > 0:
                self.last_frame_time = time.time()
                self.consecutive_errors = 0
                self.reconnecting = False
                return frame

            # Кадр не получен
            self.consecutive_errors += 1
            time_since_last = time.time() - self.last_frame_time
            
            # Проверяем, не пора ли переподключаться
            if self.consecutive_errors > 5:  # Снизил порог для быстрой реакции
                if not self.reconnecting:
                    print(f"[WARN] Камера {self.actual_index}: ошибок чтения: {self.consecutive_errors}")
                
                # Пробуем мягкий сброс (re-read без полного reconnect)
                for _ in range(3):
                    time.sleep(0.1)
                    ret2, frame2 = self.cap.read()
                    if ret2 and frame2 is not None and frame2.size > 0:
                        self.consecutive_errors = 0
                        self.last_frame_time = time.time()
                        return frame2
                
                # Мягкий сброс не помог — полный reconnect
                if self.consecutive_errors > 10:
                    print(f"[WARN] Камера {self.actual_index}: запускаем полное переподключение...")
                    if self.reconnect():
                        ret3, frame3 = self.cap.read()
                        if ret3 and frame3 is not None and frame3.size > 0:
                            self.consecutive_errors = 0
                            self.last_frame_time = time.time()
                            return frame3
                    return None
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Ошибка чтения камеры {self.actual_index}: {e}")
            self.consecutive_errors += 1
            return None

    def reconnect(self) -> bool:
        print(f"[INFO] === ПЕРЕПОДКЛЮЧЕНИЕ камеры {self.original_index} (текущий индекс: {self.actual_index}) ===")
        
        # Освобождаем текущий индекс
        if self.actual_index is not None:
            CameraManager._used_indices.discard(self.actual_index)
            print(f"[INFO] Индекс {self.actual_index} освобожден")
        
        # Закрываем старый VideoCapture
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        self.is_opened = False
        self.actual_index = None
        
        # Ждём, пока система увидит переподключенное устройство
        print(f"[INFO] Ожидание {self.retry_delay}с перед переподключением...")
        time.sleep(self.retry_delay)
        
        for attempt in range(self.max_retries):
            print(f"[INFO] Попытка {attempt + 1}/{self.max_retries}")
            
            # Пробуем открыть (open() сам найдёт свободный индекс)
            if self.open():
                self.reconnecting = False
                self.consecutive_errors = 0
                self.last_frame_time = time.time()
                print(f"[INFO] === Камера {self.original_index} успешно переподключена на индексе {self.actual_index} ===")
                return True
            
            if attempt < self.max_retries - 1:
                wait = self.retry_delay * (attempt + 1)
                print(f"[INFO] Ожидание {wait}с...")
                time.sleep(wait)
        
        print(f"[ERROR] === Не удалось переподключить камеру {self.original_index} после {self.max_retries} попыток ===")
        self.reconnecting = False
        return False

    def get_status(self):
        return {
            "is_opened": self.is_opened,
            "reconnecting": self.reconnecting,
            "consecutive_errors": self.consecutive_errors,
            "time_since_last_frame": time.time() - self.last_frame_time,
            "backend": self._backend_name(self.backend) if self.backend else "None",
            "actual_index": self.actual_index,
            "original_index": self.original_index
        }

    def release(self):
        print(f"[INFO] Освобождение камеры {self.original_index} (индекс: {self.actual_index})")
        if self.actual_index is not None:
            CameraManager._used_indices.discard(self.actual_index)
        
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        self.is_opened = False
        self.reconnecting = False
        self.actual_index = None

    def __del__(self):
        self.release()


def list_cameras(max_test=10):
    """Возвращает список индексов доступных камер"""
    available = []
    
    for idx in range(max_test):
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available.append(idx)
                    print(f"[INFO] Найдена камера {idx}: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    # Камера открылась, но не дает кадры - тоже считаем
                    print(f"[INFO] Камера {idx}: открыта, но нет кадра")
                    # available.append(idx)  # Раскомментируйте, если хотите учитывать такие камеры
            cap.release()
        except Exception as e:
            print(f"[DEBUG] Камера {idx}: {e}")
    
    return available


def count_cameras(max_test=10):
    """Подсчитывает количество доступных камер"""
    cams = list_cameras(max_test)
    return len(cams)


class CameraDetector:
    """Автоматическое определение камер при запуске"""
    
    def __init__(self, max_cameras=10):
        self.max_cameras = max_cameras
        self.available_indices = []
        self.count = 0
        
    def scan(self):
        """Сканирует доступные камеры"""
        print("[INFO] Сканирование доступных камер...")
        self.available_indices = list_cameras(self.max_cameras)
        self.count = len(self.available_indices)
        
        if self.count == 0:
            print("[WARN] Камеры не найдены!")
        else:
            print(f"[INFO] Найдено камер: {self.count}")
            print(f"[INFO] Индексы камер: {self.available_indices}")
        
        return self.count
    
    def get_camera_indices(self):
        """Возвращает индексы доступных камер"""
        return self.available_indices