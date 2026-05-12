import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import SmallUNet
from .dataset import EdgeDataset, get_transforms
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# ── Loss functions ────────────────────────────────────────────────────────────

def dice_loss(pred, target, eps=1e-6):
    pred   = torch.sigmoid(pred)
    target = target.float()
    num    = 2 * (pred * target).sum(dim=(2, 3))
    den    = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1 - (num + eps) / den).mean()


def combined_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    return bce * 0.5 + dice_loss(pred, target)


def boundary_iou(pred_mask, true_mask):
    pred_mask = pred_mask.astype(np.uint8)
    true_mask = true_mask.astype(np.uint8)
    k      = np.ones((3, 3), np.uint8)
    pred_b = cv2.morphologyEx(pred_mask, cv2.MORPH_GRADIENT, k)
    true_b = cv2.morphologyEx(true_mask, cv2.MORPH_GRADIENT, k)
    inter  = np.logical_and(pred_b, true_b).sum()
    union  = np.logical_or(pred_b,  true_b).sum()
    return inter / (union + 1e-6)


# ── Trainer Class & Helper ───────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self):
        run_stage3_training(
            train_input=self.cfg['train_input'],
            train_target=self.cfg['train_target'],
            val_input=self.cfg['val_input'],
            val_target=self.cfg['val_target'],
            checkpoint_dir=self.cfg['checkpoint_dir'],
            img_size=self.cfg.get('img_size', 512),
            batch_size=self.cfg.get('batch_size', 24),
            lr=self.cfg.get('lr', 3e-3),
            epochs=self.cfg.get('epochs', 60),
            base_c=self.cfg.get('base_c', 64),
            csv_path=self.cfg.get('csv_path', None)
        )

def default_cfg():
    return {
        'train_input': './data/train/input',
        'train_target': './data/train/target',
        'val_input': './data/val/input',
        'val_target': './data/val/target',
        'checkpoint_dir': './checkpoints',
        'img_size': 512,
        'batch_size': 24,
        'lr': 3e-3,
        'epochs': 60,
        'base_c': 64,
        'csv_path': None
    }


# ── Main training function ─────────────────────────────────────────────────────

def run_stage3_training(
    train_input,
    train_target,
    val_input,
    val_target,
    checkpoint_dir,
    img_size=512,
    batch_size=24,
    lr=3e-3,
    epochs=60,
    base_c=64,
    csv_path=None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallUNet(base_c=base_c)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer  = optim.Adam(model.parameters(), lr=lr)
    train_t, val_t = get_transforms(img_size)

    # csv_path is forwarded to EdgeDataset to filter unusable images
    train_ds = EdgeDataset(train_input, train_target, transform=train_t, csv_path=csv_path)
    val_ds   = EdgeDataset(val_input,   val_target,   transform=val_t,   csv_path=csv_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    best_val_loss = float("inf")
    save_path     = os.path.join(checkpoint_dir, "best_unet.pth")
    log           = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train")

        for imgs, masks in pbar:
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            imgs, masks = imgs.to(device).float(), masks.to(device).float()
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_ds)

        model.eval()
        val_loss     = 0
        biou_scores  = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                imgs, masks = imgs.to(device).float(), masks.to(device).float()
                preds      = model(imgs)
                loss       = combined_loss(preds, masks)
                val_loss  += loss.item() * imgs.size(0)
                preds_sig  = torch.sigmoid(preds)
                pred_bin   = (preds_sig > 0.5).float()
                for i in range(imgs.size(0)):
                    p = pred_bin[i].cpu().numpy()[0]
                    t = masks[i].cpu().numpy()[0]
                    biou_scores.append(boundary_iou(p, t))

        val_loss  /= len(val_ds)
        mean_biou  = float(np.mean(biou_scores))
        log.append([epoch, train_loss, val_loss, mean_biou])
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BIoU: {mean_biou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            inner = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(inner.state_dict(), save_path)
            print(f"--> Saved best model (Val Loss: {val_loss:.4f}) → {save_path}")

    # ── Save logs & curves ────────────────────────────────────────────────────
    df = pd.DataFrame(log, columns=["Epoch", "TrainLoss", "ValLoss", "BIoU"])
    df.to_csv(os.path.join(checkpoint_dir, "training_metrics.csv"), index=False)

    plt.figure()
    plt.plot(df["Epoch"], df["TrainLoss"], label="Train Loss")
    plt.plot(df["Epoch"], df["ValLoss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Stage 3 – Training vs Validation Loss")
    plt.legend(); plt.savefig(os.path.join(checkpoint_dir, "training_curve.png"), dpi=300); plt.close()

    plt.figure()
    plt.plot(df["Epoch"], df["BIoU"], "x-", label="BIoU")
    plt.xlabel("Epoch"); plt.ylabel("BIoU")
    plt.title("Stage 3 – Boundary IoU Across Epochs")
    plt.legend(); plt.savefig(os.path.join(checkpoint_dir, "biou_curve.png"), dpi=300); plt.close()

    return save_path


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 Training")
    parser.add_argument("--train_input",   required=True)
    parser.add_argument("--train_target",  required=True)
    parser.add_argument("--val_input",     required=True)
    parser.add_argument("--val_target",    required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--csv_path",      default=None, help="Stage 1 CSV to filter unusable images")
    parser.add_argument("--img_size",      type=int,   default=512)
    parser.add_argument("--batch_size",    type=int,   default=24)
    parser.add_argument("--lr",            type=float, default=3e-3)
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--base_c",        type=int,   default=64)
    args = parser.parse_args()

    run_stage3_training(
        train_input   = args.train_input,
        train_target  = args.train_target,
        val_input     = args.val_input,
        val_target    = args.val_target,
        checkpoint_dir= args.checkpoint_dir,
        img_size      = args.img_size,
        batch_size    = args.batch_size,
        lr            = args.lr,
        epochs        = args.epochs,
        base_c        = args.base_c,
        csv_path      = args.csv_path,
    )
