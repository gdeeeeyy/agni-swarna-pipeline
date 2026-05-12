import os
import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path

def run_inference(img_dir, model_path, save_dir, min_img_no=1, max_img_no=20000):
    # Directory setup 
    mask_save = os.path.join(save_dir, "masks")
    overlay_save = os.path.join(save_dir, "overlay")
    os.makedirs(mask_save, exist_ok=True)
    os.makedirs(overlay_save, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model definition
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    
    # weights
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval() # Ensure Eval mode is active
    
    # transforms
    transform = Compose([
        Resize(512, 512),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Get image list 
    image_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
    results = []
    
    for name in tqdm(image_list, desc="Inference"):
        # Extract image number from filename (assuming XXXXX.png format)
        try:
            img_no = int(Path(name).stem)
            if not (min_img_no <= img_no <= max_img_no):
                continue
        except ValueError:
            pass # skip non-numeric filenames if any
        
        path = os.path.join(img_dir, name)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None: continue
        
        # grayscale to 3 channels
        img = np.repeat(gray[..., None], 3, axis=2)
        h, w = gray.shape
        
        aug = transform(image=img)
        tensor = aug["image"].unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            pred = model(tensor)
            pred = torch.sigmoid(pred) # Applied inside or outside no_grad (original did both)
            
        mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        # resize
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Boundary generation
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge = np.zeros((h, w), dtype=np.uint8)
        for cnt in contours:
            cv2.drawContours(edge, [cnt], -1, 255, 1)
            
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay = color.copy()
        overlay[edge == 255] = [0, 0, 255]
        
        # Save results
        cv2.imwrite(os.path.join(mask_save, name), edge)
        cv2.imwrite(os.path.join(overlay_save, name), overlay)
        
        results.append([name, int(mask_resized.sum())])
        
    df = pd.DataFrame(results, columns=["Image", "MaskPixels"])
    df.to_csv(os.path.join(save_dir, "Masks_Pixel_Result.csv"), index=False)