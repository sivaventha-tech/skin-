"""
============================================================
SKIN DISEASE CLASSIFICATION
PROPOSED ATTENTION-ENHANCED EFFICIENTNET-B0
============================================================

Proposed architecture:

Input Image
    ↓
EfficientNet-B0 Backbone
    ↓
Channel Attention
    ↓
Spatial Attention
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Fully Connected Layer
    ↓
22 Skin Disease Classes

Attention mechanism:
    CBAM-style Channel Attention
    CBAM-style Spatial Attention

============================================================
"""

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


# ============================================================
# CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        hidden_channels = max(
            channels // reduction,
            1
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.max_pool = nn.AdaptiveMaxPool2d(
            1
        )

        self.mlp = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_attention = self.mlp(
            self.avg_pool(x)
        )

        max_attention = self.mlp(
            self.max_pool(x)
        )

        attention = (
            avg_attention
            + max_attention
        )

        attention = self.sigmoid(
            attention
        )

        return x * attention


# ============================================================
# SPATIAL ATTENTION
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(
        self,
        kernel_size=7
    ):

        super().__init__()

        padding = (
            kernel_size - 1
        ) // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_attention = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_attention, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined = torch.cat(
            [
                avg_attention,
                max_attention
            ],
            dim=1
        )

        attention = self.conv(
            combined
        )

        attention = self.sigmoid(
            attention
        )

        return x * attention


# ============================================================
# PROPOSED ATTENTION MODULE
# ============================================================

class AttentionModule(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        self.channel_attention = (
            ChannelAttention(
                channels=channels,
                reduction=reduction
            )
        )

        self.spatial_attention = (
            SpatialAttention(
                kernel_size=7
            )
        )

    def forward(self, x):

        x = self.channel_attention(
            x
        )

        x = self.spatial_attention(
            x
        )

        return x


# ============================================================
# PROPOSED MODEL
# ============================================================

class ProposedModel(nn.Module):

    def __init__(
        self,
        num_classes=22,
        pretrained=True,
        freeze_backbone=True,
        dropout=0.4
    ):

        super().__init__()

        # ----------------------------------------------------
        # EfficientNet-B0 backbone
        # ----------------------------------------------------

        if pretrained:

            weights = (
                EfficientNet_B0_Weights.DEFAULT
            )

        else:

            weights = None

        backbone = efficientnet_b0(
            weights=weights
        )

        # ----------------------------------------------------
        # EfficientNet feature extractor
        # ----------------------------------------------------

        self.features = backbone.features

        # ----------------------------------------------------
        # Freeze backbone
        # ----------------------------------------------------

        if freeze_backbone:

            for parameter in (
                self.features.parameters()
            ):

                parameter.requires_grad = False

        # ----------------------------------------------------
        # EfficientNet-B0 final feature channels
        # ----------------------------------------------------

        self.feature_channels = 1280

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = AttentionModule(
            channels=self.feature_channels,
            reduction=16
        )

        # ----------------------------------------------------
        # Global pooling
        # ----------------------------------------------------

        self.global_pool = (
            nn.AdaptiveAvgPool2d(
                output_size=1
            )
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                self.feature_channels,
                512
            ),

            nn.BatchNorm1d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                256,
                num_classes
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        # EfficientNet feature extraction
        x = self.features(
            x
        )

        # Attention refinement
        x = self.attention(
            x
        )

        # Global average pooling
        x = self.global_pool(
            x
        )

        # Classification
        x = self.classifier(
            x
        )

        return x


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_information(
    model
):

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    print()
    print("=" * 80)
    print("PROPOSED MODEL INFORMATION")
    print("=" * 80)

    print()

    print(
        "Model:"
    )

    print(
        "Attention-Enhanced EfficientNet-B0"
    )

    print()

    print(
        f"Number of classes : 22"
    )

    print(
        f"Input size        : 224 x 224"
    )

    print(
        f"Feature channels  : 1280"
    )

    print()

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen parameters    : "
        f"{frozen_parameters:,}"
    )

    print()

    print(
        "Attention:"
    )

    print(
        "✓ Channel Attention"
    )

    print(
        "✓ Spatial Attention"
    )

    print()

    print(
        "Backbone:"
    )

    print(
        "✓ EfficientNet-B0"
    )

    print(
        "✓ ImageNet pretrained"
    )

    print(
        "✓ Backbone frozen"
    )


# ============================================================
# TEST MODEL
# ============================================================

def test_model():

    print()
    print("=" * 80)
    print("PROPOSED MODEL TEST")
    print("=" * 80)

    print()

    model = ProposedModel(
        num_classes=22,
        pretrained=True,
        freeze_backbone=True
    )

    model.eval()

    print_model_information(
        model
    )

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    )

    print()
    print("=" * 80)
    print("TESTING FORWARD PASS")
    print("=" * 80)

    print()

    print(
        f"Input shape  : "
        f"{tuple(dummy_input.shape)}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print(
        f"Output shape : "
        f"{tuple(output.shape)}"
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    expected_shape = (
        2,
        22
    )

    if tuple(output.shape) == expected_shape:

        print()

        print(
            "✓ Forward pass successful"
        )

        print(
            "✓ Output shape is correct"
        )

        print(
            "✓ Proposed model test passed"
        )

    else:

        print()

        print(
            "✗ ERROR: Incorrect output shape"
        )

        print(
            f"Expected: {expected_shape}"
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
        print("PROPOSED MODEL TEST FAILED")
        print("=" * 80)

        print()

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise