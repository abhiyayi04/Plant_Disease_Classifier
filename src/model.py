from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_n_blocks(model: nn.Module, n: int = 2) -> None:
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.features.parameters():
        param.requires_grad = False

    if n <= 0:
        return

    total_blocks = len(model.features)
    start = max(0, total_blocks - n)

    for idx in range(start, total_blocks):
        for param in model.features[idx].parameters():
            param.requires_grad = True


def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]