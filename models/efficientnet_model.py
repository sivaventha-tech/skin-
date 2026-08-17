"""
============================================================
SKIN DISEASE CLASSIFICATION
EFFICIENTNET-B0 TRANSFER LEARNING MODEL
============================================================
"""

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class EfficientNetModel(nn.Module):

    def __init__(
        self,
        num_classes=22,
        pretrained=True,
        freeze_backbone=True
    ):

        super().__init__()

        # ----------------------------------------------------
        # Load EfficientNet-B0
        # ----------------------------------------------------

        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        self.backbone = efficientnet_b0(
            weights=weights
        )

        # ----------------------------------------------------
        # Freeze backbone
        # ----------------------------------------------------

        if freeze_backbone:

            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = False

        # ----------------------------------------------------
        # Original classifier
        #
        # EfficientNet-B0:
        # classifier[1] = Linear(1280, 1000)
        # ----------------------------------------------------

        input_features = (
            self.backbone.classifier[-1].in_features
        )

        # ----------------------------------------------------
        # Replace ImageNet classifier
        # ----------------------------------------------------

        self.backbone.classifier = nn.Sequential(

            nn.Dropout(
                p=0.40
            ),

            nn.Linear(
                input_features,
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.30
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
    print("EFFICIENTNET-B0 MODEL TEST")
    print("=" * 80)

    print()

    model = EfficientNetModel(
        num_classes=22,
        pretrained=True,
        freeze_backbone=True
    )

    model.eval()

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            dummy_input
        )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if tuple(output.shape) == (2, 22):

        print(
            "✓ EfficientNet-B0 model test passed"
        )

    else:

        print(
            "✗ EfficientNet-B0 output shape is incorrect"
        )

        print(
            f"Expected: (2, 22)"
        )

        print(
            f"Received: {tuple(output.shape)}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        test_model()

    except Exception as error:

        print()
        print("=" * 80)
        print("EFFICIENTNET MODEL TEST FAILED")
        print("=" * 80)

        print()

        print(
            f"{type(error).__name__}: {error}"
        )

        raise