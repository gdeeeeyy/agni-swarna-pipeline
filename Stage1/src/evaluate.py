import torch
from torch.utils.data import DataLoader
from src.dataset import ImageTagDataset
from src.transforms import get_transforms
from src.model import get_resnet34


def evaluate(csv_path, flame_dir, model_path, batch_size=16):
    """
    Evaluate a trained model on the given dataset.

    Args:
        csv_path (str): Path to CSV file
        flame_dir (str): Directory containing images
        model_path (str): Path to trained model (.pth)
        batch_size (int): Batch size for evaluation
    """

    dataset = ImageTagDataset(csv_path, flame_dir, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet34(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"Final Accuracy: {acc:.2f}%")