import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from pathlib import Path
from .dataset import BaselineDataset
from .evaluate import metrics

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_usable_ids(csv_path, min_img_no=19501, max_img_no=20000):
    df = pd.read_csv(csv_path)
    label_col = "PredictedLabel" if "PredictedLabel" in df.columns else "ImageTag"
    df[label_col] = df[label_col].astype(str).str.lower()
    df = df[df[label_col] == "usable"]
    df = df[(df["ImageNo"] >= min_img_no) & (df["ImageNo"] <= max_img_no)]
    ids = sorted(df["ImageNo"].astype(int).tolist())
    print(f"\n--- Data Loader Info ---")
    print(f"Using column: {label_col}")
    print(f"Target Range: {min_img_no} to {max_img_no}")
    print(f"Found {len(ids)} usable images for training.")
    print(f"------------------------\n")
    return ids

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=8.0, bce_w=0.75):
        super().__init__()
        self.bce_w = bce_w
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def dice_loss(self, logits, target, eps=1e-6):
        p = torch.sigmoid(logits)
        num = 2.0 * (p * target).sum(dim=(2, 3))
        den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
        return 1.0 - (num / den).mean()

    def forward(self, logits, target):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight.to(logits.device)
        )
        return self.bce_w * bce + (1.0 - self.bce_w) * self.dice_loss(logits, target)

def run_training(img_dir, mask_dir, csv_path, save_dir, batch_size, epochs, lr, weight_decay, pos_weight, bce_weight, num_workers, seed, min_img_no=19501, max_img_no=20000):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = get_usable_ids(csv_path, min_img_no, max_img_no)
    if not ids:
        raise ValueError(f"No usable images found in the 19501-20000 range in {csv_path}!")

    dataset = BaselineDataset(img_dir=img_dir, mask_dir=mask_dir, image_ids=ids, augment=True)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=RandomSampler(train_ds), num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model definition - using online ImageNet weights
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    
    model = model.to(device)
    criterion = BCEDiceLoss(pos_weight, bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "edge_model_1.pth")

    best_biou, log = 0.0, []

    for ep in tqdm(range(1, epochs + 1), desc="Stage 2 Training"):
        model.train()
        epoch_loss = 0
        for x, y in tqdm(train_dl, desc=f"Epoch {ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        iou, dice, biou = metrics(model, val_dl, device)
        log.append([ep, iou, dice, biou])
        print(f"Epoch {ep:03d} | Loss: {epoch_loss/len(train_dl):.4f} | IoU: {iou:.4f} | Dice: {dice:.4f} | BIoU: {biou:.4f}")

        if biou > best_biou:
            best_biou = biou
            torch.save({"state_dict": model.state_dict()}, ckpt_path)
            print(f"--> Saved new best model (BIoU: {biou:.4f})")
            
        scheduler.step(ep)

    df_log = pd.DataFrame(log, columns=["Epoch", "IoU", "Dice", "BIoU"])
    df_log.to_csv(os.path.join(save_dir, "training_metrics.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_log["Epoch"], df_log["IoU"], "o-", label="IoU")
    plt.plot(df_log["Epoch"], df_log["Dice"], "s-", label="Dice")
    plt.plot(df_log["Epoch"], df_log["BIoU"], "x-", label="BIoU")
    plt.title("Performance vs Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_curve_biou.png"), dpi=300)
    plt.close()

    return ckpt_path