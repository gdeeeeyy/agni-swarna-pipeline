import torch.nn as nn
from torchvision import models

def get_resnet34(num_classes=2, pretrained=True):
    weights = "DEFAULT" if pretrained else None
    model = models.resnet34(weights=weights)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model