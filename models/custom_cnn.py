"""
============================================================
SKIN DISEASE CLASSIFICATION
CUSTOM CNN BASELINE MODEL
============================================================

Input:
    224 x 224 RGB image

Output:
    22 skin disease classes

Architecture:
    Block 1: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
             -> ReLU -> MaxPool -> Dropout

    Block 2: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
             -> ReLU -> MaxPool -> Dropout

    Block 3: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
             -> ReLU -> MaxPool -> Dropout

    Block 4: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
             -> ReLU -> MaxPool -> Dropout

    Classifier:
        Adaptive Average Pooling
        Fully Connected
        Dropout
        22-class output

============================================================
"""

import torch
import torch.nn as nn


# ============================================================
# CUSTOM CNN
# ============================================================

class CustomCNN(nn.Module):

    def __init__(
        self,
        num_classes=22
    ):

        super().__init__()

        self.num_classes = num_classes

        # ====================================================
        # CONVOLUTIONAL BLOCK 1
        # ====================================================

        self.block1 = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Dropout2d(
                p=0.10
            )
        )


        # ====================================================
        # CONVOLUTIONAL BLOCK 2
        # ====================================================

        self.block2 = nn.Sequential(

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Dropout2d(
                p=0.15
            )
        )


        # ====================================================
        # CONVOLUTIONAL BLOCK 3
        # ====================================================

        self.block3 = nn.Sequential(

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Dropout2d(
                p=0.20
            )
        )


        # ====================================================
        # CONVOLUTIONAL BLOCK 4
        # ====================================================

        self.block4 = nn.Sequential(

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Dropout2d(
                p=0.25
            )
        )


        # ====================================================
        # ADAPTIVE POOLING
        # ====================================================

        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                in_features=256,
                out_features=128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=0.50
            ),

            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
        )


        # ====================================================
        # INITIALIZE WEIGHTS
        # ====================================================

        self._initialize_weights()


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(
        self,
        x
    ):

        x = self.block1(
            x
        )

        x = self.block2(
            x
        )

        x = self.block3(
            x
        )

        x = self.block4(
            x
        )

        x = self.global_pool(
            x
        )

        x = self.classifier(
            x
        )

        return x


    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    def _initialize_weights(
        self
    ):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if module.bias is not None:

                    nn.init.constant_(
                        module.bias,
                        0
                    )

            elif isinstance(
                module,
                nn.BatchNorm2d
            ):

                nn.init.constant_(
                    module.weight,
                    1
                )

                nn.init.constant_(
                    module.bias,
                    0
                )

            elif isinstance(
                module,
                nn.Linear
            ):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01
                )

                nn.init.constant_(
                    module.bias,
                    0
                )


# ============================================================
# MODEL SUMMARY
# ============================================================

def print_model_summary(
    num_classes=22
):

    model = CustomCNN(
        num_classes=num_classes
    )

    print(
        "=" * 70
    )

    print(
        "CUSTOM CNN MODEL"
    )

    print(
        "=" * 70
    )

    print()

    print(
        model
    )

    print()

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
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print()

    print(
        "=" * 70
    )


# ============================================================
# MODEL TEST
# ============================================================

def test_model():

    print()
    print(
        "Testing CustomCNN..."
    )

    model = CustomCNN(
        num_classes=22
    )

    model.eval()

    # Dummy batch:
    # 2 images
    # 3 RGB channels
    # 224 x 224 pixels

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

    print(
        f"Input shape  : "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"Output shape : "
        f"{tuple(output.shape)}"
    )

    expected_shape = (
        2,
        22
    )

    if tuple(output.shape) == expected_shape:

        print(
            "✓ Model test passed"
        )

    else:

        print(
            "✗ Model output shape is incorrect"
        )

        print(
            f"Expected: {expected_shape}"
        )

        print(
            f"Got     : {tuple(output.shape)}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_model_summary(
        num_classes=22
    )

    test_model()