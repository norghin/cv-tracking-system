# web/api_server.py
import json
import asyncio
from aiohttp import web

class TrackingAPI:
    def __init__(self, cam_states, cam_locks, cameras, control_params, reid_model, video_streamer):
        self.cam_states = cam_states
        self.cam_locks = cam_locks
        self.cameras = cameras
        self.control_params = control_params
        self.reid_model = reid_model
        self.video_streamer = video_streamer
        
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        from web.video_streamer import WebSocketHandler, broadcast_loop
        
        ws_handler = WebSocketHandler(self.video_streamer)
        
        # WebSocket
        self.app.router.add_get('/ws', ws_handler.handle)
        
        # Главная страница
        self.app.router.add_get('/', self.index)
        
        # Статические файлы
        self.app.router.add_static('/static', 'web/static')
        
        # API
        self.app.router.add_get('/api/status', self.api_status)
        self.app.router.add_get('/api/params', self.api_get_params)
        self.app.router.add_post('/api/params', self.api_set_params)
        
        # Запускаем broadcast при старте
        async def start_broadcast(app):
            asyncio.create_task(broadcast_loop(self.video_streamer, app))
        
        self.app.on_startup.append(start_broadcast)
    
    async def index(self, request):
        return web.FileResponse('web/static/dashboard.html')
    
    async def api_status(self, request):
        running = sum(1 for cid in self.cam_states if self.cam_states[cid].running)
        tracks = sum(
            self.cam_states[cid].metrics.tracks[-1] if self.cam_states[cid].metrics.tracks else 0
            for cid in self.cam_states
        )
        return web.json_response({
            "active_cameras": running,
            "total_cameras": len(self.cam_states),
            "total_tracks": tracks,
            "reid_active": self.reid_model.get_active_count() if self.reid_model else 0
        })
    
    async def api_get_params(self, request):
        """Возвращает текущие параметры."""
        params = {}
        for k, v in self.control_params.__dict__.items():
            if not k.startswith('_'):
                params[k] = v
        
        # Добавляем ReID параметры
        if self.reid_model:
            params.update({
                "reid_similarity_threshold": self.reid_model.similarity_threshold,
                "reid_strong_threshold": self.reid_model.strong_threshold,
                "reid_max_age": self.reid_model.max_age,
                "reid_max_gallery_size": self.reid_model.max_gallery_size,
                "reid_alpha": self.reid_model.alpha
            })
        
        return web.json_response(params)
    
    async def api_set_params(self, request):
        """Устанавливает параметры."""
        try:
            data = await request.json()
            
            # ReID параметры
            reid_params = {
                "reid_similarity_threshold": "similarity_threshold",
                "reid_strong_threshold": "strong_threshold",
                "reid_max_age": "max_age",
                "reid_max_gallery_size": "max_gallery_size",
                "reid_alpha": "alpha"
            }
            
            for key, value in data.items():
                if key in reid_params and self.reid_model:
                    setattr(self.reid_model, reid_params[key], value)
                elif hasattr(self.control_params, key):
                    setattr(self.control_params, key, value)
            
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
    
    def run(self, host='0.0.0.0', port=8000):
        print(f"[API] Starting on http://{host}:{port}")
        web.run_app(self.app, host=host, port=port, print=lambda x: None)