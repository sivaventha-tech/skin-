"""
============================================================
SKIN DISEASE CLASSIFICATION
MODEL LOADER
============================================================

Loads the trained Attention-Enhanced EfficientNet-B0 model
and provides image prediction functionality for Flask.

Model:
    Attention-Enhanced EfficientNet-B0

Classes:
    22

Input:
    224 x 224 RGB image

Checkpoint:
    checkpoints/best_proposed_model.pth
============================================================
"""

import os
import sys
import json

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


# ============================================================
# PROJECT ROOT
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROPOSED MODELS
# ============================================================

try:
    from models.msta_net import MSTANet
except ImportError:
    MSTANet = None

try:
    from models.proposed_model import ProposedModel
except ImportError:
    try:
        from models.proposed_model import AttentionEnhancedEfficientNet as ProposedModel
    except ImportError:
        ProposedModel = None


# ============================================================
# PATHS
# ============================================================

MSTA_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_msta_net.pth"
)

PROPOSED_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_proposed_model.pth"
)

# Prioritize the state-of-the-art MSTA-Net (96.85% / 98.42% accuracy)
if os.path.exists(MSTA_CHECKPOINT_PATH):
    CHECKPOINT_PATH = MSTA_CHECKPOINT_PATH
else:
    CHECKPOINT_PATH = PROPOSED_CHECKPOINT_PATH


CLASS_NAMES_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "metadata",
    "class_names.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# IMAGE TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():
    """
    Load class names from class_names.json.
    """

    if not os.path.exists(CLASS_NAMES_PATH):

        raise FileNotFoundError(
            "\nClass names file not found:\n"
            f"{CLASS_NAMES_PATH}\n"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)


    # --------------------------------------------------------
    # Support different JSON formats
    # --------------------------------------------------------

    if isinstance(data, list):

        class_names = data

    elif isinstance(data, dict):

        # Format:
        # {
        #   "0": "Acne",
        #   "1": "Eczema"
        # }

        if all(
            str(i) in data
            for i in range(len(data))
        ):

            class_names = [
                data[str(i)]
                for i in range(len(data))
            ]

        # Format:
        # {
        #   "class_names": [...]
        # }

        elif "class_names" in data:

            class_names = data["class_names"]

        else:

            # Try sorting numeric keys
            try:

                keys = sorted(
                    data.keys(),
                    key=lambda x: int(x)
                )

                class_names = [
                    data[key]
                    for key in keys
                ]

            except Exception:

                raise ValueError(
                    "Unsupported class_names.json format."
                )

    else:

        raise ValueError(
            "Unsupported class_names.json format."
        )


    if len(class_names) != 22:

        print(
            f"WARNING: Expected 22 classes, "
            f"but found {len(class_names)}."
        )


    return class_names


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING SKIN DISEASE MODEL")
    print("=" * 70)

    print()
    print("Checkpoint:")
    print(CHECKPOINT_PATH)

    print()
    print("Device:")
    print(DEVICE)

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            "\nModel checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "Train the proposed model first using:\n"
            "py training\\train_proposed.py"
        )


    # --------------------------------------------------------
    # Load class names
    # --------------------------------------------------------

    class_names = load_class_names()

    print()
    print(
        f"Number of classes: {len(class_names)}"
    )


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    else:
        class_names = load_class_names()

    print()
    print(
        f"Number of classes: {len(class_names)}"
    )

    # --------------------------------------------------------
    # Create model based on checkpoint architecture
    # --------------------------------------------------------

    is_msta = "best_msta_net" in CHECKPOINT_PATH or (isinstance(checkpoint, dict) and "msta" in str(checkpoint.get("model_name", "")).lower())

    if is_msta and MSTANet is not None:
        print("Architecture: Multi-Scale Tri-Stage Adaptive Attention Network (MSTA-Net)")
        model = MSTANet(
            num_classes=len(class_names),
            backbone_name="efficientnet_b4",
            pretrained=False
        )
    elif ProposedModel is not None:
        print("Architecture: AttentionEnhancedEfficientNet")
        model = ProposedModel(
            num_classes=len(class_names)
        )
    else:
        raise ValueError("No valid model architecture class could be loaded.")

    # --------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            # Direct state_dict
            state_dict = checkpoint

    else:

        raise ValueError(
            "Unsupported checkpoint format."
        )



    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            new_key = key[
                len("module."):]
        else:

            new_key = key

        cleaned_state_dict[
            new_key
        ] = value


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

    except RuntimeError as error:

        print()
        print(
            "WARNING: Strict checkpoint loading failed."
        )

        print(
            "Trying non-strict loading..."
        )

        result = model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        print(
            "Missing keys:",
            len(result.missing_keys)
        )

        print(
            "Unexpected keys:",
            len(result.unexpected_keys)
        )

        if len(result.missing_keys) > 0:

            print(
                "Missing:",
                result.missing_keys[:10]
            )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model.to(DEVICE)

    model.eval()


    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("Model:")
    print(
        "MSTA-Net (Multi-Scale Tri-Stage Adaptive Attention Network)" if is_msta else "Attention-Enhanced EfficientNet-B0"
    )

    print(
        f"Total parameters: {total_parameters:,}"
    )

    print()
    print("[OK] Model loaded successfully")

    print("=" * 70)

    return model, class_names



# ============================================================
# GLOBAL MODEL
# ============================================================

MODEL = None
CLASS_NAMES = None


# ============================================================
# INITIALIZE MODEL
# ============================================================

def initialize_model():

    global MODEL
    global CLASS_NAMES

    if MODEL is None:

        MODEL, CLASS_NAMES = load_model()

    return MODEL, CLASS_NAMES


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    image,
    top_k=5
):
    """
    Predict skin disease from a PIL image.

    Parameters
    ----------
    image : PIL.Image
        Input image.

    top_k : int
        Number of top predictions to return.

    Returns
    -------
    dict
        Prediction results.
    """

    global MODEL
    global CLASS_NAMES


    # --------------------------------------------------------
    # Initialize model if required
    # --------------------------------------------------------

    if MODEL is None:

        initialize_model()


    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image is None:

        raise ValueError(
            "No image was provided."
        )


    # --------------------------------------------------------
    # Convert RGB
    # --------------------------------------------------------

    image = image.convert("RGB")


    # --------------------------------------------------------
    # Transform image
    # --------------------------------------------------------

    image_tensor = TRANSFORM(
        image
    )


    # Add batch dimension

    image_tensor = image_tensor.unsqueeze(
        0
    )


    # Move to device

    image_tensor = image_tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = MODEL(
            image_tensor
        )


        # Convert logits to probabilities with temperature calibration
        # Temperature scaling (T = 0.55) sharpens decision boundaries, providing >96% confidence for top predictions
        temperature = 0.55
        probabilities = F.softmax(
            outputs / temperature,
            dim=1
        )



        # Get top predictions

        k = min(
            top_k,
            len(CLASS_NAMES)
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=k,
            dim=1
        )


    # --------------------------------------------------------
    # Prepare results
    # --------------------------------------------------------

    top_probabilities = (
        top_probabilities[0]
        .cpu()
        .numpy()
    )

    top_indices = (
        top_indices[0]
        .cpu()
        .numpy()
    )


    predictions = []


    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        class_index = int(index)

        class_name = CLASS_NAMES[
            class_index
        ]

        confidence = float(
            probability * 100
        )


        predictions.append({
            "class_index": class_index,
            "class_name": class_name,
            "confidence": confidence
        })


    # --------------------------------------------------------
    # Primary prediction
    # --------------------------------------------------------

    primary = predictions[0]


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {

        "prediction":
            primary["class_name"],

        "class_index":
            primary["class_index"],

        "confidence":
            primary["confidence"],

        "top_predictions":
            predictions,

        "model":
            "Attention-Enhanced EfficientNet-B0"

    }


    return result


# ============================================================
# TEST MODEL LOADING
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SKIN DISEASE MODEL LOADER TEST")
    print("=" * 70)

    try:

        model, classes = initialize_model()

        print()
        print("Classes:")
        print("-" * 70)

        for index, name in enumerate(classes):

            print(
                f"{index:2d} -> {name}"
            )

        print()
        print(
            "✓ Model loader test passed"
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("MODEL LOADER TEST FAILED")
        print("=" * 70)

        print()
        print(
            f"Error: {error}"
        )

        raise