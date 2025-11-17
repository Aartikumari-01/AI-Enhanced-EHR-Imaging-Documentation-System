from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
from PIL import Image
import shutil

app = FastAPI(title="Breast Ultrasound EHR AI Integration")

# Serve static HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure required folders exist
os.makedirs("ehr_integration", exist_ok=True)
os.makedirs("enhanced_results", exist_ok=True)
os.makedirs("enhanced_outputs", exist_ok=True)

# -----------------------------
# ICD MAPPING RULES
# -----------------------------
ICD_RULES = {
    "breast cancer": "C50.3",
    "tumor": "C50.3",
    "lump": "C50.3",

    # Endocrine
    "diabetes": "E11",
    "thyroid": "E03.9",
    "hypothyroidism": "E03.9",
    "hyperthyroidism": "E05.9",

    # Respiratory
    "asthma": "J45",
    "pneumonia": "J18.9",
    "bronchitis": "J20.9",
    "copd": "J44.9",
    "tuberculosis": "A15.0",

    # Cardiovascular
    "hypertension": "I10",
    "high blood pressure": "I10",
    "heart attack": "I21",
    "myocardial infarction": "I21",
    "heart failure": "I50.9",
    "arrhythmia": "I49.9",

    # Neurology
    "stroke": "I63.9",
    "epilepsy": "G40.9",
    "headache": "R51",

    # Gastrointestinal
    "gastritis": "K29.7",
    "ulcer": "K25.9",
    "liver disease": "K76.9",
    "hepatitis": "B19.9",

    # Renal
    "kidney failure": "N17.9",
    "uti": "N39.0",
    "urinary infection": "N39.0",

    # Musculoskeletal
    "arthritis": "M19.90",
    "back pain": "M54.5",
    "fracture": "S52.90",

    # Infection / Fever
    "fever": "R50.9",
    "viral infection": "B34.9",
    "bacterial infection": "A49.9",

    # OBGYN
    "pregnancy": "Z34.90",
    "pcos": "E28.2",
    "fibroid": "D25.9",

    # Dermatology
    "skin infection": "L08.9",
    "dermatitis": "L30.9",

    # Mental health
    "depression": "F32.9",
    "anxiety": "F41.9",
}

DEFAULT_ICD = "C50.3"


# -----------------------------------------
# ICD Prediction
# -----------------------------------------
def predict_icd10_from_notes(text: str):
    text = text.lower()
    for keyword, code in ICD_RULES.items():
        if keyword in text:
            return code
    return DEFAULT_ICD


# -----------------------------------------
# Note Generator
# -----------------------------------------
def make_note(age, gender, doctor_notes, icd_code):
    return (
        f"{age}-year-old {gender} patient shows clinical signs related to ICD-10 "
        f"code {icd_code}. Summary of symptoms: {doctor_notes}"
    )


# -----------------------------------------
# Response Model
# -----------------------------------------
class PredictResponse(BaseModel):
    patient_id: str
    generated_note: str
    predicted_icd_code: str
    saved_to_ehr: bool


# -----------------------------------------
# MAIN PREDICTION API
# -----------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    patient_id: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    doctor_notes: str = Form(...),
    image: UploadFile = File(None)
):

    # -----------------------------
    # IMAGE VALIDATION (OPTIONAL)
    # -----------------------------
    img_path = None

    if image and image.filename != "":
        allowed = (".png", ".jpg", ".jpeg", ".dcm")

        if not image.filename.lower().endswith(allowed):
            return {"error": "Only medical image formats allowed: PNG, JPG, JPEG, DCM"}

        # Save original image
        img_path = os.path.join("enhanced_outputs", f"{patient_id}_{image.filename}")
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Try to make a preview image
        try:
            img = Image.open(img_path).convert("L").resize((224, 224))
            preview_path = os.path.join("enhanced_outputs", f"preview_{patient_id}.png")
            img.save(preview_path)
        except Exception:
            pass

    # -----------------------------
    # ICD PREDICTION
    # -----------------------------
    predicted_code = predict_icd10_from_notes(doctor_notes)

    # Generate clinical note
    generated_note = make_note(age, gender, doctor_notes, predicted_code)

    # -----------------------------
    # Save EHR Record
    # -----------------------------
    ehr_record = {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "doctor_notes": doctor_notes,
        "image_path": img_path,
        "generated_note": generated_note,
        "predicted_icd_code": predicted_code
    }

    with open(f"ehr_integration/{patient_id}.json", "w") as f:
        json.dump(ehr_record, f, indent=2)

    return PredictResponse(
        patient_id=patient_id,
        generated_note=generated_note,
        predicted_icd_code=predicted_code,
        saved_to_ehr=True
    )


# -----------------------------------------
# EHR FETCH
# -----------------------------------------
@app.get("/ehr/{patient_id}")
def get_ehr(patient_id: str):
    file_path = f"ehr_integration/{patient_id}.json"
    if not os.path.exists(file_path):
        return JSONResponse({"error": "Patient not found"}, status_code=404)
    return json.load(open(file_path))
