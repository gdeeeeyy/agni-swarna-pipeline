import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = (512, 512)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class BaselineDataset(Dataset):
    def __init__(self, img_dir, image_ids, mask_dir=None, augment=True):
        self.img_dir = Path(img_dir)
        self.image_ids = image_ids
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.use_mask = self.mask_dir is not None
        self.tf = self._build_tf(augment and self.use_mask)

    def _build_tf(self, augment):
        t = []
        if augment:
            t += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(0.05, 0.05, 10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ]
        t += [A.Resize(*IMG_SIZE), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
        return A.Compose(t)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = str(self.image_ids[idx]).zfill(5)
        img_path = self.img_dir / f"{img_id}.png"
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: raise RuntimeError(f"Missing image {img_id}")
        img = np.repeat(img[..., None], 3, axis=2)

        if not self.use_mask:
            out = self.tf(image=img)
            return out["image"], img_id

        mask_path = self.mask_dir / f"{img_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: raise RuntimeError(f"Missing mask {img_id}")

        mask = (mask > 127).astype("uint8")
        # Exact original dilation
        k = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)

        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0).float()