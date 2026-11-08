import asyncio
import json
import base64
import cv2
import time
from aiohttp import web

class VideoStreamer:
    def __init__(self, cam_states, cam_locks, cameras, control_params, reid_model):
        self.cam_states = cam_states
        self.cam_locks = cam_locks
        self.cameras = cameras
        self.control_params = control_params
        self.reid_model = reid_model
        
        self.clients = set()
        self.jpeg_quality = 35
        self.max_fps = 10
        
        self.last_frames = {}
        self.last_frame_time = {}
        self._running = True
        self.frame_cache = {}
        self.last_frame_hash = {}
    
    def add_client_sync(self, ws):
        self.clients.add(ws)
        print(f"[WS] Client connected. Total: {len(self.clients)}")
    
    def remove_client_sync(self, ws):
        self.clients.discard(ws)
        print(f"[WS] Client disconnected. Total: {len(self.clients)}")
    
    def get_frame_data(self):
        frames_data = {}
        metrics_data = {}
        
        for cam_id in self.cam_states:
            try:
                with self.cam_locks[cam_id]:
                    state = self.cam_states[cam_id]
                    camera = self.cameras[cam_id]
                    
                    if not state.running:
                        continue
                    
                    frame = state.display_frame if state.display_frame is not None else state.frame
                    
                    if frame is None or frame.size == 0:
                        continue
                    
                    now = time.time()
                    
                    # Проверяем FPS-лимит
                    if cam_id in self.last_frame_time:
                        if now - self.last_frame_time[cam_id] < 1.0 / self.max_fps:
                            # Используем кешированный кадр, если он есть
                            if cam_id in self.frame_cache:
                                frames_data[str(cam_id)] = self.frame_cache[cam_id]
                                continue
                    
                    # Вычисляем хеш кадра (простой, но быстрый)
                    frame_hash = hash(frame.tobytes()) if hasattr(frame, 'tobytes') else id(frame)
                    
                    # Если кадр не изменился и есть в кеше — используем кеш
                    if (cam_id in self.last_frame_hash and 
                        frame_hash == self.last_frame_hash[cam_id] and 
                        cam_id in self.frame_cache):
                        
                        self.last_frame_time[cam_id] = now
                        frames_data[str(cam_id)] = self.frame_cache[cam_id]
                        
                    else:
                        # Кадр изменился — кодируем заново
                        self.last_frame_time[cam_id] = now
                        self.last_frame_hash[cam_id] = frame_hash
                        
                        # Ресайз если нужно
                        h, w = frame.shape[:2]
                        target_h = 360
                        scale = target_h / h if h > target_h else 1.0
                        
                        if scale < 1.0:
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            frame_small = cv2.resize(frame, (new_w, new_h))
                        else:
                            frame_small = frame.copy()
                        
                        # Кодируем в JPEG
                        ret, jpg = cv2.imencode('.jpg', frame_small, 
                                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                        if ret:
                            jpg_b64 = base64.b64encode(jpg).decode('utf-8')
                            frames_data[str(cam_id)] = jpg_b64
                            self.frame_cache[cam_id] = jpg_b64
                            self.last_frames[cam_id] = jpg_b64
                    
                    # Метрики (легковесные, не кешируем)
                    cam_status = camera.get_status()
                    
                    line_counts = None
                    if state.line_enabled:
                        line_counts = state.line_counter.get_counts()
                    
                    metrics_data[str(cam_id)] = {
                        "fps": round(state.metrics.fps[-1], 1) if state.metrics.fps else 0,
                        "tracks": state.metrics.tracks[-1] if state.metrics.tracks else 0,
                        "avg_age": round(state.metrics.avg_age[-1], 1) if state.metrics.avg_age else 0,
                        "reconnecting": cam_status.get("reconnecting", False),
                        "errors": cam_status.get("consecutive_errors", 0),
                        "line_counts": line_counts
                    }
                    
            except Exception as e:
                print(f"[WS] Camera {cam_id} error: {e}")
        
        system_metrics = {
            "reid_active": self.reid_model.get_active_count() if self.reid_model else 0,
            "total_cameras": len(self.cam_states)
        }
        
        return json.dumps({
            "type": "frame_update",
            "frames": frames_data,
            "metrics": metrics_data,
            "system": system_metrics
        })
    
    def stop(self):
        self._running = False


class WebSocketHandler:
    def __init__(self, streamer):
        self.streamer = streamer
        self.msg_count = 0  # Счётчик сообщений
    
    async def handle(self, request):
        ws = web.WebSocketResponse(compress=15)
        await ws.prepare(request)
        
        self.streamer.add_client_sync(ws)
        print(f"[WS] ✓ Client connected. Total: {len(self.streamer.clients)}")
        
        try:
            info = json.dumps({
                "type": "system_info",
                "cameras": len(self.streamer.cam_states),
                "reid_active": self.streamer.reid_model.get_active_count() if self.streamer.reid_model else 0
            })
            await ws.send_str(info)
        except Exception as e:
            print(f"[WS] ⚠ Failed to send system_info: {e}")
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    self.msg_count += 1
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        # ============ SET_PARAM ============
                        if msg_type == "set_param":
                            param = data.get("param")
                            value = data.get("value")
                            
                            reid_params = {
                                "reid_similarity_threshold": "similarity_threshold",
                                "reid_strong_threshold": "strong_threshold", 
                                "reid_max_age": "max_age",
                                "reid_max_gallery_size": "max_gallery_size",
                                "reid_alpha": "alpha"
                            }
                            
                            if param in reid_params and self.streamer.reid_model:
                                old_val = getattr(self.streamer.reid_model, reid_params[param])
                                setattr(self.streamer.reid_model, reid_params[param], value)
                                new_val = getattr(self.streamer.reid_model, reid_params[param])
                                print(f"[WS] REID PARAM #{self.msg_count}: {param} = {old_val} → {new_val}")
                                
                            elif param and hasattr(self.streamer.control_params, param):
                                old_val = getattr(self.streamer.control_params, param)
                                setattr(self.streamer.control_params, param, value)
                                new_val = getattr(self.streamer.control_params, param)
                                print(f"[WS] CONTROL PARAM #{self.msg_count}: {param} = {old_val} → {new_val}")
                            else:
                                print(f"[WS] ⚠ UNKNOWN PARAM #{self.msg_count}: {param} = {value} (not found in reid_model or control_params)")
                        
                        # ============ GET_PARAMS ============
                        elif msg_type == "get_params":
                            params = {}
                            if self.streamer.reid_model:
                                params.update({
                                    "reid_similarity_threshold": self.streamer.reid_model.similarity_threshold,
                                    "reid_strong_threshold": self.streamer.reid_model.strong_threshold,
                                    "reid_max_age": self.streamer.reid_model.max_age,
                                    "reid_max_gallery_size": self.streamer.reid_model.max_gallery_size,
                                    "reid_alpha": self.streamer.reid_model.alpha
                                })
                            await ws.send_str(json.dumps({"type": "params_update", "params": params}))
                            print(f"[WS] GET_PARAMS #{self.msg_count}: sent {len(params)} params")
                        
                        # ============ CLEAR_REID ============
                        elif msg_type == "clear_reid":
                            if self.streamer.reid_model:
                                with self.streamer.reid_model.lock:
                                    old_count = len(self.streamer.reid_model.embeddings)
                                    self.streamer.reid_model.embeddings.clear()
                                    self.streamer.reid_model.last_seen.clear()
                                    self.streamer.reid_model.avg_boxes.clear()
                                    self.streamer.reid_model.next_global_id = 1
                                await ws.send_str(json.dumps({"type": "reid_cleared"}))
                                print(f"[WS] CLEAR_REID #{self.msg_count}: removed {old_count} embeddings")
                        
                        # ============ TOGGLE_LINE ============
                        elif msg_type == "toggle_line":
                            cam_id = data.get("cam_id", 0)
                            enabled = data.get("enabled", True)
                            if cam_id in self.streamer.cam_states:
                                old_state = self.streamer.cam_states[cam_id].line_enabled
                                self.streamer.cam_states[cam_id].line_enabled = enabled
                                print(f"[WS] TOGGLE_LINE #{self.msg_count}: cam={cam_id} {old_state} → {enabled}")
                            else:
                                print(f"[WS] ⚠ TOGGLE_LINE #{self.msg_count}: cam={cam_id} not found")
                        
                        # ============ RESET_COUNTS ============
                        elif msg_type == "reset_counts":
                            cam_id = data.get("cam_id", 0)
                            if cam_id in self.streamer.cam_states:
                                old_counts = self.streamer.cam_states[cam_id].line_counter.get_counts()
                                self.streamer.cam_states[cam_id].line_counter.reset()
                                print(f"[WS] RESET_COUNTS #{self.msg_count}: cam={cam_id}, was: in={old_counts['in']}, out={old_counts['out']}")
                            else:
                                print(f"[WS] ⚠ RESET_COUNTS #{self.msg_count}: cam={cam_id} not found")
                        
                        # ============ SET_LINE ============
                        elif msg_type == "set_line":
                            cam_id = data.get("cam_id", 0)
                            line_type = data.get("line_type", "straight")
                            points = data.get("points", [[100, 240], [540, 240]])
                            if cam_id in self.streamer.cam_states:
                                state = self.streamer.cam_states[cam_id]
                                old_pts = state.line_counter.points
                                state.line_counter.set_line(line_type, points)
                                state.line_enabled = True
                                print(f"[WS] SET_LINE #{self.msg_count}: cam={cam_id}, type={line_type}")
                                print(f"    old: {old_pts}")
                                print(f"    new: {state.line_counter.points}")
                            else:
                                print(f"[WS] ⚠ SET_LINE #{self.msg_count}: cam={cam_id} not found")
                        
                        # ============ ROTATE_LINE ============
                        elif msg_type == "rotate_line":
                            cam_id = int(data.get("cam_id", 0))
                            angle = int(data.get("angle", 0))
                            if cam_id in self.streamer.cam_states:
                                state = self.streamer.cam_states[cam_id]
                                old_angle = state.line_counter.current_angle
                                state.line_counter.rotate(angle)
                                print(f"[WS] ROTATE #{self.msg_count}: cam={cam_id}, angle: {old_angle}° → {angle}°")
                                # Логируем точки только если они изменились
                                if old_angle != angle:
                                    print(f"    points: {state.line_counter.points}")
                            else:
                                print(f"[WS] ⚠ ROTATE #{self.msg_count}: cam={cam_id} not found")
                        
                        # ============ GET_COUNTS ============
                        elif msg_type == "get_counts":
                            cam_id = data.get("cam_id", 0)
                            if cam_id in self.streamer.cam_states:
                                counts = self.streamer.cam_states[cam_id].line_counter.get_counts()
                                await ws.send_str(json.dumps({
                                    "type": "counts_update", 
                                    "cam_id": cam_id, 
                                    "counts": counts
                                }))
                                print(f"[WS] GET_COUNTS #{self.msg_count}: cam={cam_id}, in={counts['in']}, out={counts['out']}, inside={counts['inside']}")
                        
                        # ============ UNKNOWN ============
                        else:
                            print(f"[WS] ❓ UNKNOWN MSG #{self.msg_count}: type={msg_type}, data={data}")
                    
                    except json.JSONDecodeError as e:
                        print(f"[WS] ❌ JSON ERROR #{self.msg_count}: {e}")
                    except Exception as e:
                        print(f"[WS] ❌ HANDLER ERROR #{self.msg_count}: {e} | data={msg.data[:200]}")
                
                elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                    print(f"[WS] ✕ Client disconnecting (msg type: {msg.type})")
                    break
                    
        except Exception as e:
            print(f"[WS] ⚠ Handler error: {e}")
        finally:
            self.streamer.remove_client_sync(ws)
            print(f"[WS] ✕ Client disconnected. Total: {len(self.streamer.clients)}")
        
        return ws


async def broadcast_loop(streamer, app):
    print("[WS] Broadcast loop STARTED")
    count = 0
    
    while streamer._running:
        if not streamer.clients:
            await asyncio.sleep(0.1)
            continue
        
        count += 1
        message = streamer.get_frame_data()  # Это строка JSON
        
        if count % 100 == 0:  # Логируем реже
            msg_size = len(message)
            print(f"[WS] Sent {count} messages to {len(streamer.clients)} clients, size: {msg_size} bytes")
        
        dead = set()
        for ws in list(streamer.clients):
            try:
                await ws.send_str(message)  # Компрессия работает автоматически
            except:
                dead.add(ws)
        
        for ws in dead:
            streamer.remove_client_sync(ws)
        
        await asyncio.sleep(1.0 / streamer.max_fps)
    
    print("[WS] Broadcast loop STOPPED")