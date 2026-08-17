"""
============================================================
SKIN DISEASE CLASSIFICATION
RESNET50 TRANSFER LEARNING MODEL
============================================================
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Model(nn.Module):

    def __init__(
        self,
        num_classes=22,
        pretrained=True,
        freeze_backbone=True
    ):

        super().__init__()

        # ----------------------------------------------------
        # Load ImageNet pretrained ResNet50
        # ----------------------------------------------------

        if pretrained:

            weights = ResNet50_Weights.DEFAULT

        else:

            weights = None

        self.backbone = resnet50(
            weights=weights
        )

        # ----------------------------------------------------
        # Freeze backbone
        # ----------------------------------------------------

        if freeze_backbone:

            for parameter in self.backbone.parameters():

                parameter.requires_grad = False

        # ----------------------------------------------------
        # Replace original ImageNet classifier
        # ----------------------------------------------------

        input_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Sequential(

            nn.Linear(
                input_features,
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.40
            ),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# TEST MODEL
# ============================================================

def test_model():

    print("=" * 80)
    print("RESNET50 MODEL TEST")
    print("=" * 80)

    print()

    model = ResNet50Model(
        num_classes=22,
        pretrained=True,
        freeze_backbone=True
    )

    model.eval()

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    )

    with torch.no_grad():

        output = model(
            dummy_input
        )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Input shape          : "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"Output shape         : "
        f"{tuple(output.shape)}"
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print()

    if tuple(output.shape) == (2, 22):

        print(
            "✓ ResNet50 model test passed"
        )

    else:

        print(
            "✗ ResNet50 output shape incorrect"
        )


if __name__ == "__main__":

    test_model()