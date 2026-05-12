import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from .model import SmallUNet
from torch.utils.data import Dataset, DataLoader

# Dataset 
class EdgeDataset(Dataset):
    def __init__(self, input_dir, target_dir, img_size=256):
        self.input_files  = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".png")])
        self.target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith(".png")])
        self.img_size = img_size

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.input_files[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.target_files[idx], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return img, mask


# Metrics
def dice_coef(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return (num / den).item()

def iou_coef(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection + eps
    return (intersection / union).item()


# Evaluation
def evaluate(model_path, input_dir, target_dir, img_size=256, batch_size=8, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Load model
    model = SmallUNet(in_ch=1, out_ch=1, base_c=64)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # Dataset & loader
    ds = EdgeDataset(input_dir, target_dir, img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    dice_scores = []
    iou_scores = []

    for imgs, masks in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            preds = torch.sigmoid(model(imgs))

        for p, t in zip(preds, masks):
            dice_scores.append(dice_coef(p, t))
            iou_scores.append(iou_coef(p, t))

    print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU : {np.mean(iou_scores):.4f}")


# Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--target", required=True, help="Target masks folder")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    evaluate(args.model, args.input, args.target, args.img_size, args.batch_size)
