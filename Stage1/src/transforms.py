import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
 
def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])