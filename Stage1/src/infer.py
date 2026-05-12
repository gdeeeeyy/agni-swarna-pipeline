import os
import re
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])


def get_resnet34(num_classes=2, pretrained=False):
    model = models.resnet34(pretrained=pretrained)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_inference(
    img_dir,
    save_dir,
    model_path,
    min_img_no=1,
    max_img_no=20000
):
    img_dir = os.path.abspath(img_dir)
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    output_csv = os.path.join(save_dir, "final_usable_unusable_images.csv")
    skipped_csv = os.path.join(save_dir, "skipped_images_00001_20000.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_resnet34(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    transform = get_val_transforms()

    image_files = sorted(
        f for f in os.listdir(img_dir)
        if re.fullmatch(r"\d{5}\.png", f)
    )

    results = []
    skipped = []

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Stage1 Inference"):
            img_no = int(os.path.splitext(img_file)[0])

            if not (min_img_no <= img_no <= max_img_no):
                continue

            img_path = os.path.join(img_dir, img_file)

            try:
                img = np.array(Image.open(img_path).convert("L"))
            except Exception as e:
                skipped.append({
                    "ImageFile": img_file,
                    "Error": str(e)
                })
                continue

            img_tensor = (
                transform(image=img)["image"]
                .unsqueeze(0)
                .to(device)
            )

            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())

            results.append({
                "ImageNo": os.path.splitext(img_file)[0],
                "PredictedLabel": "Usable" if pred == 0 else "Unusable"
            })

    print(f"Inference complete. Saving {len(results)} results to {output_csv}...")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    
    if os.path.exists(output_csv):
        print(f"Successfully saved Stage 1 results to {output_csv}")
    else:
        print(f"ERROR: Failed to save Stage 1 results to {output_csv}")

    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_csv, index=False)

    print(
        f"Saved predictions for {len(results)} valid images "
        f"({min_img_no:05d}–{max_img_no:05d}) to {output_csv}"
    )

    if skipped:
        print(
            f"Skipped {len(skipped)} invalid images "
            f"(see {skipped_csv})"
        )


if __name__ == "__main__":
    IMG_DIR = "/home/tanvir/Desktop/01-08-2026/Stage1/data/output_baseline"
    SAVE_DIR = "/home/tanvir/Desktop/01-08-2026/Stage1/results"
    BEST_MODEL_PATH = (
        "/home/tanvir/Desktop/01-08-2026/Stage1/results/models/"
        "best_model_fold6.pth"
    )

    run_inference(
        img_dir=IMG_DIR,
        save_dir=SAVE_DIR,
        model_path=BEST_MODEL_PATH
    )