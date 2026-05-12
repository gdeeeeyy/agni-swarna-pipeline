import segmentation_models_pytorch as smp

def get_model(device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
    return model
