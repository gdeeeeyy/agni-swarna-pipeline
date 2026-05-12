import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from src.transforms import get_transforms, get_val_transforms
from src.dataset import ImageTagDataset
from src.model import get_resnet34
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm


DEFAULT_CSV_PATH = "data/final_cleaned_image_sequence.csv"
DEFAULT_IMG_DIR = "data/output_baseline"
DEFAULT_SAVE_DIR = "results"

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_KFOLDS = 10
DEFAULT_SEED = 42


def train_kfold(
    csv_path=DEFAULT_CSV_PATH,
    img_dir=DEFAULT_IMG_DIR,
    save_dir=DEFAULT_SAVE_DIR,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    kfolds=DEFAULT_KFOLDS,
    seed=DEFAULT_SEED,
):
    csv_path = Path(csv_path)
    img_dir = Path(img_dir)
    save_dir = Path(save_dir)

    (save_dir / "plots").mkdir(parents=True, exist_ok=True)
    (save_dir / "models").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = ImageTagDataset(
        csv_path=str(csv_path),
        img_dir=str(img_dir),
        transform=get_transforms()
    )

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

    fold_results = []
    global_best_acc = -1.0
    global_best_model_path = None

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(full_dataset)), total=kfolds, desc="KFolds"):
        print(f"\n" + "="*30)
        print(f" Fold {fold + 1}/{kfolds} ")
        print("="*30)

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        model = get_resnet34(num_classes=2, pretrained=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_acc = -1.0
        best_model_path = save_dir / "models" / f"best_model_fold{fold + 1}.pth"
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []

            for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)

            model.eval()
            val_losses, preds, targets = [], [], []

            with torch.no_grad():
                for imgs, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_losses.append(loss.item())
                    preds.extend(outputs.argmax(1).cpu().numpy())
                    targets.extend(labels.cpu().numpy())

            val_loss = np.mean(val_losses)
            val_acc = accuracy_score(targets, preds)
            val_prec = precision_score(targets, preds, average="macro", zero_division=0)
            val_rec = recall_score(targets, preds, average="macro", zero_division=0)
            val_f1 = f1_score(targets, preds, average="macro", zero_division=0)

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss={train_loss:.4f} | "
                f"Val Loss={val_loss:.4f} | "
                f"Val Acc={val_acc:.4f} | "
                f"P={val_prec:.4f} R={val_rec:.4f} F1={val_f1:.4f}"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

        fold_results.append(best_acc)

        if best_acc > global_best_acc:
            global_best_acc = best_acc
            global_best_model_path = best_model_path

    # Copy the global best fold model to save_dir/best_model.pth
    best_model_final = save_dir / "best_model.pth"
    shutil.copy2(global_best_model_path, best_model_final)
    print(f"Best model saved to: {best_model_final}")

        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Usable", "Unusable"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix (Fold {fold + 1})")
        plt.savefig(save_dir / "plots" / f"cm_fold{fold + 1}.png")
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title(f"Loss (Fold {fold + 1})")

        plt.subplot(1, 2, 2)
        plt.plot(history["val_acc"], label="Val Accuracy")
        plt.legend()
        plt.title(f"Val Acc (Fold {fold + 1})")

        plt.tight_layout()
        plt.savefig(save_dir / "plots" / f"train_curve_fold{fold + 1}.png")
        plt.close()

    model = get_resnet34(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(global_best_model_path, map_location=device, weights_only=False))
    model.eval()

    transform = get_val_transforms()
    df = pd.read_csv(csv_path)

    results = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Final Predictions"):
            img_path = img_dir / f"{str(row['ImageNo']).zfill(5)}.png"
            img = np.array(Image.open(img_path).convert("L"))
            img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            results.append({
                "ImageNo": row["ImageNo"],
                "ImageTag": row["ImageTag"],
                "PredictedLabel": "Usable" if pred == 0 else "Unusable",
                "Probability_Usable": probs[0],
                "Probability_Unusable": probs[1],
            })

    output_csv = save_dir / "final_usable_unusable_images.csv"
    print(f"Training-phase inference complete. Saving {len(results)} results to {output_csv}...")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    
    if output_csv.exists():
        print(f"Successfully saved Stage 1 training-phase results to {output_csv}")
    else:
        print(f"ERROR: Failed to save Stage 1 training-phase results to {output_csv}")

    print(f"\nSaved predictions to {output_csv}")
    print("\n=== Final Results ===")
    for i, acc in enumerate(fold_results, 1):
        print(f"Fold {i}: {acc:.4f}")
    print(f"Average Accuracy: {np.mean(fold_results):.4f}")
    print(f"Selected model: {global_best_model_path}")
    print(f"Best model (for pipeline): {best_model_final}")

    return str(best_model_final)


if __name__ == "__main__":
    train_kfold()