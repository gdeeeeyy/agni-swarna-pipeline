import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = (512,512)
IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def get_train_tf():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(0.05,0.05,10,p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Resize(*IMG_SIZE),
        A.Normalize(IMAGENET_MEAN,IMAGENET_STD),
        ToTensorV2()
    ])

def get_val_tf():
    return A.Compose([
        A.Resize(*IMG_SIZE),
        A.Normalize(IMAGENET_MEAN,IMAGENET_STD),
        ToTensorV2()
    ])
