"""
============================================================
SKIN DISEASE CLASSIFICATION
PREDICTION ROUTE
============================================================

Handles:
    POST /predict

Receives:
    Uploaded skin image

Returns:
    result.html with prediction results
============================================================
"""

import os
import sys

from flask import (
    Blueprint,
    request,
    render_template,
    redirect,
    url_for,
    flash
)

from PIL import Image


# ============================================================
# PROJECT ROOT
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORT MODEL LOADER
# ============================================================

from web.model_loader import predict_image


# ============================================================
# BLUEPRINT
# ============================================================

predict_bp = Blueprint(
    "predict",
    __name__
)


# ============================================================
# ALLOWED EXTENSIONS
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# CHECK FILE EXTENSION
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = (
        filename
        .rsplit(".", 1)[1]
        .lower()
    )

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# PREDICTION ROUTE
# ============================================================

@predict_bp.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if "image" not in request.files:

        flash(
            "Please select a skin image."
        )

        return redirect(
            url_for("home")
        )


    file = request.files["image"]


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        flash(
            "No image was selected."
        )

        return redirect(
            url_for("home")
        )


    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        flash(
            "Invalid image format. "
            "Use JPG, JPEG, PNG, or WEBP."
        )

        return redirect(
            url_for("home")
        )


    # --------------------------------------------------------
    # Check file size
    # --------------------------------------------------------

    file.seek(
        0,
        os.SEEK_END
    )

    file_size = file.tell()

    file.seek(0)


    if file_size > MAX_FILE_SIZE:

        flash(
            "Image is too large. "
            "Maximum size is 10 MB."
        )

        return redirect(
            url_for("home")
        )


    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            file
        )

        image.load()

        image = image.convert(
            "RGB"
        )

    except Exception as error:

        print(
            f"Image loading error: {error}"
        )

        flash(
            "The uploaded file is not a valid image."
        )

        return redirect(
            url_for("home")
        )


    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    try:

        result = predict_image(
            image,
            top_k=5
        )

    except Exception as error:

        print()
        print(
            "=" * 70
        )

        print(
            "PREDICTION ERROR"
        )

        print(
            "=" * 70
        )

        print(
            error
        )

        print(
            "=" * 70
        )

        flash(
            "Unable to analyze the image. "
            "Please check the model checkpoint."
        )

        return redirect(
            url_for("home")
        )


    # ========================================================
    # PREPARE RESULT DATA
    # ========================================================

    prediction = result[
        "prediction"
    ]

    confidence = result[
        "confidence"
    ]

    top_predictions = result[
        "top_predictions"
    ]

    model_name = result[
        "model"
    ]


    # --------------------------------------------------------
    # Image preview
    #
    # Convert uploaded image to a data URL so that we don't
    # need to permanently store patient images on the server.
    # --------------------------------------------------------

    import base64
    from io import BytesIO


    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )


    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


    image_data_url = (
        "data:image/jpeg;base64,"
        + image_base64
    )


    # ========================================================
    # PRINT RESULT TO TERMINAL
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "SKIN DISEASE PREDICTION"
    )

    print(
        "=" * 70
    )

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    print(
        f"Model      : {model_name}"
    )

    print()
    print(
        "Top Predictions:"
    )

    for index, item in enumerate(
        top_predictions,
        start=1
    ):

        print(
            f"{index}. "
            f"{item['class_name']} "
            f"- "
            f"{item['confidence']:.2f}%"
        )

    print(
        "=" * 70
    )


    # ========================================================
    # RENDER RESULT PAGE
    # ========================================================

    return render_template(
        "result.html",

        prediction=prediction,

        confidence=confidence,

        top_predictions=top_predictions,

        model_name=model_name,

        image_data=image_data_url
    )


# ============================================================
# FILE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 70
    )

    print(
        "PREDICTION ROUTE"
    )

    print(
        "=" * 70
    )

    print(
        "Blueprint name : predict"
    )

    print(
        "Route          : POST /predict"
    )

    print(
        "Allowed files  : JPG, JPEG, PNG, WEBP"
    )

    print(
        "Maximum size   : 10 MB"
    )

    print(
        "=" * 70
    )

    print(
        "✓ Prediction route configuration loaded"
    )