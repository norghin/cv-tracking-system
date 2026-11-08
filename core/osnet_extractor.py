import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchreid

class OSNetExtractor:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)

        self.model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=1000,
            pretrained=True
        ).to(self.device)

        self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = (256, 128)  # (H, W)

    def preprocess(self, roi):
        if roi is None or roi.size == 0:
            return None
        img = cv2.resize(roi, self.input_size[::-1])
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def extract(self, roi):
        tensor = self.preprocess(roi)
        if tensor is None:
            return None
        feat = self.model(tensor)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy()[0]

    @torch.no_grad()
    def extract_batch(self, rois):
        """
        rois: список BGR‑картинок (вырезанных людей)
        Возвращает список эмбеддингов (512D) для каждой ROI
        """
        tensors = []
        for roi in rois:
            t = self.preprocess(roi)
            if t is not None:
                tensors.append(t)

        if not tensors:
            return []

        batch = torch.cat(tensors, dim=0)  # объединяем в один батч
        feats = self.model(batch)
        feats = F.normalize(feats, p=2, dim=1)

        return feats.cpu().numpy()

