import time
import numpy as np
import torch
import torch.nn.functional as F
import threading

class GlobalReID:
    """
    Глобальный менеджер идентификаций с галереей эмбеддингов + EMA + двойной порог.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.50, 
        strong_threshold: float = 0.65,      
        max_age: float = 60.0,
        max_gallery_size: int = 10,
        alpha: float = 0.4,
        cleanup_interval: int = 100
    ):
        self.similarity_threshold = similarity_threshold
        self.strong_threshold = strong_threshold
        self.max_age = max_age
        self.max_gallery_size = max_gallery_size
        self.alpha = alpha
        self.cleanup_interval = cleanup_interval

        self.next_global_id = 1
        self.embeddings = {}
        self.last_seen = {}
        self.avg_boxes = {}
        self._calls = 0
        self.lock = threading.Lock()

    def _cleanup(self):
        now = time.time()
        with self.lock:
            to_delete = [gid for gid, ts in self.last_seen.items() if now - ts > self.max_age]
            for gid in to_delete:
                self.embeddings.pop(gid, None)
                self.last_seen.pop(gid, None)
                self.avg_boxes.pop(gid, None)

    def _ema_update(self, old_emb, new_emb):
        old_emb = np.array(old_emb, dtype=np.float32)
        new_emb = np.array(new_emb, dtype=np.float32)
        return self.alpha * new_emb + (1 - self.alpha) * old_emb

    def _best_similarity(self, emb, gallery):
        """Вычисляет максимальное косинусное сходство с галереей."""
        arr = np.array(gallery[-self.max_gallery_size:], dtype=np.float32)
        emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        gallery_t = torch.from_numpy(arr)
        sims = F.cosine_similarity(emb_t, gallery_t)
        return float(torch.max(sims))

    def register(self, emb: np.ndarray, bbox: tuple = None) -> int:
        gid = self.next_global_id
        self.next_global_id += 1
        emb = np.array(emb, dtype=np.float32)
        with self.lock:
            self.embeddings[gid] = [emb]
            self.last_seen[gid] = time.time()
            if bbox is not None:
                _, _, w, h = bbox
                self.avg_boxes[gid] = (w, h)
        print(f"[ReID] NEW global ID: {gid} (total: {len(self.embeddings)})")
        return gid

    def match(self, emb: np.ndarray):
        """Ищет лучший match в галерее."""
        with self.lock:
            if not self.embeddings:
                return None, -1.0, {}
            embeddings_copy = dict(self.embeddings)
        
        best_gid, best_sim = None, -1.0
        all_sims = {}
        
        for gid, gallery in embeddings_copy.items():
            if gallery:
                sim = self._best_similarity(emb, gallery)
                all_sims[gid] = sim
                if sim > best_sim:
                    best_sim, best_gid = sim, gid
        
        sorted_sims = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)[:3]
        if sorted_sims:
            top_str = ", ".join([f"G{gid}:{sim:.3f}" for gid, sim in sorted_sims])
            print(f"[ReID] Match query - Top: {top_str} (threshold: {self.similarity_threshold})")
        
        return best_gid, best_sim, all_sims

    def match_or_register(self, emb: np.ndarray, bbox: tuple = None) -> int:
        self._calls += 1
        if self._calls % self.cleanup_interval == 0:
            self._cleanup()

        best_gid, best_sim, all_sims = self.match(emb)

        if best_gid is not None and best_sim >= self.similarity_threshold:
            # Проверка на неоднозначность
            #second_best_sim = sorted(all_sims.values(), reverse=True)[1] if len(all_sims) > 1 else 0
            #if second_best_sim > 0 and best_sim - second_best_sim < 0.1:
            #    print(f"[ReID] Ambiguous: best={best_sim:.3f}, second={second_best_sim:.3f} - creating new ID")
            #    return self.register(emb, bbox)

            print(f"[ReID] MATCH: G{best_gid} (sim: {best_sim:.3f})")
            
            # Проверка размера бокса (оставляем как есть)
            if bbox is not None:
                with self.lock:
                    if best_gid in self.avg_boxes:
                        _, _, w, h = bbox
                        avg_w, avg_h = self.avg_boxes[best_gid]
                        if avg_w > 0 and avg_h > 0:
                            size_ratio_w = w / avg_w if avg_w > 0 else 1
                            size_ratio_h = h / avg_h if avg_h > 0 else 1
                            if size_ratio_w < 0.3 or size_ratio_w > 3.0 or size_ratio_h < 0.3 or size_ratio_h > 3.0:
                                print(f"[ReID] Size mismatch: {w}x{h} vs avg {avg_w:.0f}x{avg_h:.0f} - new ID")
                                return self.register(emb, bbox)

            with self.lock:
                if best_gid in self.embeddings:
                    gallery = self.embeddings[best_gid]
                    if gallery:
                        # === ИЗМЕНЕНИЕ 1: EMA-обновляем ВСЕ элементы галереи, а не только последний ===
                        for i in range(len(gallery)):
                            gallery[i] = self._ema_update(gallery[i], emb)
                    
                    # === ИЗМЕНЕНИЕ 2: Добавляем ВСЕГДА если есть место (не только при strong_threshold) ===
                    if len(gallery) < self.max_gallery_size:
                        gallery.append(np.array(emb, dtype=np.float32))
                        print(f"[ReID] G{best_gid}: added to gallery ({len(gallery)}/{self.max_gallery_size})")
                    # === Если галерея полная — заменяем самый старый (не последний!) ===
                    else:
                        # Циклический буфер: заменяем самый старый эмбеддинг
                        gallery.pop(0)
                        gallery.append(np.array(emb, dtype=np.float32))
                        print(f"[ReID] G{best_gid}: gallery full, replaced oldest")
                    
                    # === ИЗМЕНЕНИЕ 3: Всегда обновляем последний эмбеддинг через EMA (даже если strong_threshold не пройден) ===
                    # Это уже сделано в цикле выше
                    
                    self.last_seen[best_gid] = time.time()
                    
                    if bbox is not None:
                        _, _, w, h = bbox
                        if best_gid in self.avg_boxes:
                            avg_w, avg_h = self.avg_boxes[best_gid]
                            self.avg_boxes[best_gid] = ((avg_w + w) / 2, (avg_h + h) / 2)
                        else:
                            self.avg_boxes[best_gid] = (w, h)
            
            return best_gid

        if best_gid is not None:
            print(f"[ReID] No match: best={best_sim:.3f} < threshold={self.similarity_threshold}")
        return self.register(emb, bbox)

    def get_active_count(self):
        with self.lock:
            return len(self.embeddings)