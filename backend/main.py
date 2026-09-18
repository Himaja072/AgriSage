# ============================================
# AGRISAGE — COMPLETE FASTAPI BACKEND
# ============================================

from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from advisory.severity import estimate_severity
from advisory.risk_assessment import assess_risk
from advisory.advisory import generate_advisory


# ============================================
# PATHS
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "disease_model"

MODEL_PATH = MODEL_DIR / "best_model.keras"
LABELS_PATH = MODEL_DIR / "class_labels.json"


# ============================================
# LOAD MODEL
# ============================================

print("Loading AgriSage disease model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_data = json.load(f)

class_names = labels_data["labels"]

print("✅ Model loaded")
print(f"✅ Number of classes: {len(class_names)}")


# ============================================
# FASTAPI
# ============================================

app = FastAPI(
    title="AgriSage API",
    description="AI-Powered Crop Disease Intelligence and Early Risk Assessment System",
    version="1.0.0"
)


# ============================================
# CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# IMAGE PREPROCESSING
# ============================================

IMG_SIZE = (224, 224)


def preprocess_image(image_bytes):

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================
# ROOT
# ============================================

@app.get("/")
def root():

    return {
        "application": "AgriSage",
        "message": "AI-powered crop disease intelligence system",
        "status": "running"
    }


# ============================================
# HEALTH
# ============================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "number_of_classes": len(class_names)
    }


# ============================================
# PREDICTION
# ============================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),

    temperature: float = 25.0,
    humidity: float = 70.0,
    rainfall: float = 0.0,
    growth_stage: str = "vegetative"
):

    # ----------------------------------------
    # Validate file
    # ----------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/bmp"
    }

    if file.content_type not in allowed_types:

        return {
            "status": "error",
            "message": "Please upload a JPG, PNG, or BMP image."
        }


    # ----------------------------------------
    # Read image
    # ----------------------------------------

    image_bytes = await file.read()

    if not image_bytes:

        return {
            "status": "error",
            "message": "Uploaded image is empty."
        }


    # ----------------------------------------
    # Preprocess
    # ----------------------------------------

    image = preprocess_image(image_bytes)


    # ----------------------------------------
    # Disease prediction
    # ----------------------------------------

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    disease_name = class_names[predicted_index]


    # ----------------------------------------
    # Top 3 predictions
    # ----------------------------------------

    top_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({
            "disease": class_names[int(index)],
            "confidence": round(
                float(probabilities[index]) * 100,
                2
            )
        })


    # ----------------------------------------
    # Severity
    # ----------------------------------------

    severity_result = estimate_severity(
        disease_name,
        confidence
    )


    # ----------------------------------------
    # Risk
    # ----------------------------------------

    risk_result = assess_risk(
        disease_name=disease_name,
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall,
        growth_stage=growth_stage
    )


    # ----------------------------------------
    # Advisory
    # ----------------------------------------

    advisory_result = generate_advisory(
        disease_name=disease_name,
        severity_result=severity_result,
        risk_result=risk_result
    )


    # ----------------------------------------
    # FINAL RESPONSE
    # ----------------------------------------

    return {

        "status": "success",

        "prediction": {

            "disease": disease_name,

            "confidence": round(
                confidence * 100,
                2
            ),

            "top_predictions": top_predictions
        },

        "severity": severity_result,

        "risk_assessment": {

            "temperature": temperature,

            "humidity": humidity,

            "rainfall": rainfall,

            "growth_stage": growth_stage,

            "result": risk_result
        },

        "advisory": advisory_result
    }