"""
============================================================
SKIN DISEASE CLASSIFICATION SYSTEM
FLASK WEB APPLICATION
============================================================

Model:
    Attention-Enhanced EfficientNet-B0

Classes:
    22 Skin Disease Classes

Application:
    Upload image -> CNN prediction -> Result page
============================================================
"""

import os
import sys

from flask import (
    Flask,
    render_template,
    jsonify
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)


# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        CURRENT_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        CURRENT_DIR,
        "static"
    )
)


# ============================================================
# SECRET KEY
# ============================================================

app.config["SECRET_KEY"] = (
    "skin-disease-cnn-demo-secret-key"
)


# ============================================================
# UPLOAD CONFIGURATION
# ============================================================

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# IMPORT PREDICTION BLUEPRINT
# ============================================================

try:

    from web.predict import predict_bp

    app.register_blueprint(
        predict_bp
    )

    print(
        "✓ Prediction blueprint registered"
    )

except Exception as error:

    print()
    print("=" * 70)
    print("WARNING: PREDICTION BLUEPRINT COULD NOT BE LOADED")
    print("=" * 70)
    print(error)
    print("=" * 70)
    print()


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "running",

        "application":
            "Skin Disease Classification",

        "model":
            "Attention-Enhanced EfficientNet-B0",

        "number_of_classes":
            22,

        "input_size":
            "224x224",

        "framework":
            "PyTorch + Flask"

    })


# ============================================================
# ABOUT / MODEL INFORMATION
# ============================================================

@app.route("/model-info")
def model_info():

    return jsonify({

        "model":
            "Attention-Enhanced EfficientNet-B0",

        "backbone":
            "EfficientNet-B0",

        "attention":
            [
                "Channel Attention",
                "Spatial Attention"
            ],

        "input_size":
            "224x224",

        "classes":
            22,

        "training_type":
            "Transfer Learning",

        "explainability":
            "Grad-CAM"

    })


# ============================================================
# ERROR: FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "result.html",

        prediction="Upload Error",

        confidence=0,

        top_predictions=[],

        model_name=(
            "Attention-Enhanced "
            "EfficientNet-B0"
        ),

        image_data=None,

        error_message=(
            "The uploaded image is too large. "
            "Maximum allowed size is 10 MB."
        )
    ), 413


# ============================================================
# ERROR: INTERNAL SERVER ERROR
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    print()
    print("=" * 70)
    print("FLASK INTERNAL SERVER ERROR")
    print("=" * 70)
    print(error)
    print("=" * 70)

    return render_template(
        "result.html",

        prediction="Prediction Error",

        confidence=0,

        top_predictions=[],

        model_name=(
            "Attention-Enhanced "
            "EfficientNet-B0"
        ),

        image_data=None,

        error_message=(
            "An unexpected error occurred "
            "while processing the image."
        )
    ), 500


# ============================================================
# APPLICATION INFORMATION
# ============================================================

def print_application_info():

    print()
    print("=" * 70)

    print(
        "SKIN DISEASE CLASSIFICATION SYSTEM"
    )

    print("=" * 70)

    print()

    print(
        "Application : Flask Web Application"
    )

    print(
        "Model       : Attention-Enhanced EfficientNet-B0"
    )

    print(
        "Backbone    : EfficientNet-B0"
    )

    print(
        "Attention   : Channel + Spatial"
    )

    print(
        "Classes     : 22"
    )

    print(
        "Input       : 224 x 224 RGB"
    )

    print(
        "Framework   : PyTorch"
    )

    print()

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    print(
        "Web directory:"
    )

    print(
        CURRENT_DIR
    )

    print()

    print(
        "Home:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    print(
        "Health check:"
    )

    print(
        "http://127.0.0.1:5000/health"
    )

    print()

    print(
        "Model information:"
    )

    print(
        "http://127.0.0.1:5000/model-info"
    )

    print()

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_application_info()

    print()
    print(
        "Starting Flask development server..."
    )

    print(
        "Press CTRL+C to stop the server."
    )

    print()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True,

        use_reloader=False
    )