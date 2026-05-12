import cv2
import numpy as np
import torch

def boundary_iou(pred, true):
    pred, true = pred.astype(np.uint8), true.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    pred_b = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, k)
    true_b = cv2.morphologyEx(true, cv2.MORPH_GRADIENT, k)
    inter = np.logical_and(pred_b, true_b).sum()
    union = np.logical_or(pred_b, true_b).sum()
    return inter / (union + 1e-6)

@torch.no_grad()
def metrics(model, dataloader, device):
    model.eval()
    iou_list, dice_list, biou_list = [], [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = torch.sigmoid(logits)
        p, t = preds > 0.5, y > 0.5

        inter = (p & t).sum(dim=(1, 2, 3)).float()
        union = (p | t).sum(dim=(1, 2, 3)).float()

        iou = (inter / (union + 1e-6)).cpu().numpy()
        dice = (2 * inter / (p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + 1e-6)).cpu().numpy()

        for i in range(p.shape[0]):
            biou = boundary_iou(p[i].cpu().numpy()[0], t[i].cpu().numpy()[0])
            biou_list.append(biou)

        iou_list.extend(iou)
        dice_list.extend(dice)

    return float(np.mean(iou_list)), float(np.mean(dice_list)), float(np.mean(biou_list))