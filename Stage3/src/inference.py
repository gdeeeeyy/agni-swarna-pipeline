import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model import SmallUNet


ROOT_STAGE3 = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_STAGE3 / "checkpoints" / "best_unet.pth"

IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_stage3(input_mask_dir: str, parent_output_dir: str, model_path: str = None):
    """
    Runs Stage3 inference using Stage2 Masks.

    input_mask_dir:
        Stage2_Output/Masks

    parent_output_dir:
        results/.../Inference   (parent of Stage2_Output)
    """

    input_mask_dir = Path(input_mask_dir)
    parent_output_dir = Path(parent_output_dir)

    
    # create Stage3 output folders
    stage3_root = parent_output_dir / "Stage3_Output"
    output_dir = stage3_root / "Masks"
    overlay_dir = stage3_root / "Overlay"

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    
    # Load model
    m_path = Path(model_path) if model_path else MODEL_PATH
    model = SmallUNet(in_ch=1, out_ch=1, base_c=64)
    model.load_state_dict(torch.load(m_path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()

    input_files = sorted([f for f in os.listdir(input_mask_dir) if f.lower().endswith(".png")])

    print(f"\n[Stage3] Running on {len(input_files)} masks")
    print(f"[Stage3] Model → {m_path}")

    
    # Inference loop
    for f in tqdm(input_files, desc="Stage3 Predicting"):

        img = cv2.imread(str(input_mask_dir / f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(DEVICE)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]

        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        thin_edge = np.zeros_like(pred_mask)

        for cnt in contours:
            cv2.drawContours(thin_edge, [cnt], -1, 255, thickness=1)

        cv2.imwrite(str(output_dir / f), thin_edge)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay = img_bgr.copy()
        overlay[thin_edge == 255] = [0, 0, 255]

        cv2.imwrite(str(overlay_dir / f), overlay)

    print("\n[Stage3] Inference complete")
