import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path


class ImageTagDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.data = self.data.dropna(subset=["ImageNo", "ImageTag"])
        self.data["ImageNo"] = self.data["ImageNo"].astype(str).str.zfill(5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_dir / f"{row['ImageNo']}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = np.array(Image.open(img_path).convert("L"))
        label = 0 if row["ImageTag"].strip().lower() == "usable" else 1
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label, str(img_path)