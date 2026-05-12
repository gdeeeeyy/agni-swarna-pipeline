import os, cv2, torch, numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size):
    train_t = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.GaussNoise(var_limit=(5, 20), p=0.3),
        ToTensorV2()
    ])
    val_t = A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2()
    ])
    return train_t, val_t


def _load_usable_ids(csv_path):
    """Return a set of integer image IDs that are tagged 'usable'."""
    df = pd.read_csv(csv_path)
    label_col = "PredictedLabel" if "PredictedLabel" in df.columns else "ImageTag"
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    usable = df[df[label_col] == "usable"]["ImageNo"].astype(int).tolist()
    return set(usable)


class EdgeDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None, img_size=256, csv_path=None):
        all_inputs = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir) if f.endswith(".png")
        ])
        all_targets = sorted([
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir) if f.endswith(".png")
        ])

        if csv_path is not None:
            usable_ids = _load_usable_ids(csv_path)
            def _is_usable(path):
                try:
                    return int(os.path.splitext(os.path.basename(path))[0]) in usable_ids
                except ValueError:
                    return True  # non-numeric filenames are kept

            all_inputs  = [f for f in all_inputs  if _is_usable(f)]
            all_targets = [f for f in all_targets if _is_usable(f)]
            print(f"[EdgeDataset] CSV filter applied: {len(all_inputs)} usable samples kept.")

        self.input_files  = all_inputs
        self.target_files = all_targets
        self.transform = transform
        self.img_size  = img_size

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        img  = cv2.imread(self.input_files[idx],  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.target_files[idx], cv2.IMREAD_GRAYSCALE)

        img  = cv2.resize(img,  (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        img  = torch.tensor(img,  dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return img, mask
