import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import json
import os

# ====================================================
# 0) CHECK ENVIRONMENT
# ====================================================
print("Current working directory:", os.getcwd())
print("\n=== Clinical Note Generation & ICD-10 Automation ===\n")

# ====================================================
# 1) LOAD STRUCTURED & TEXT DATA
# ====================================================

structured_file = "structured.csv"          
unstructured_file = "unstructured_notes.csv" 
# Check if files exist
if not os.path.exists(structured_file):
    print(f"Error: {structured_file} not found in {os.getcwd()}")
    exit()
if not os.path.exists(unstructured_file):
    print(f" Error: {unstructured_file} not found in {os.getcwd()}")
    exit()

# Load CSV files
structured_df = pd.read_csv(structured_file)
unstructured_df = pd.read_csv(unstructured_file)

# Merge EHR + Doctor notes
if 'Patient_ID' not in structured_df.columns or 'Patient_ID' not in unstructured_df.columns:
    print("Error: 'Patient_ID' column missing in one of the files!")
    print(f"structured.csv columns: {structured_df.columns.tolist()}")
    print(f"unstructured_notes.csv columns: {unstructured_df.columns.tolist()}")
    exit()

clinical_df = pd.merge(structured_df, unstructured_df, on='Patient_ID')
print("\nMerged Clinical Data (Sample):")
print(clinical_df.head(), "\n")

# ====================================================
# 2) DATA CLEANING
# ====================================================

print("Cleaning data...")

clinical_df.dropna(subset=['ICD10_Code', 'Doctor_Notes', 'Age'], inplace=True)
clinical_df['Age'] = pd.to_numeric(clinical_df['Age'], errors='coerce')
clinical_df.dropna(subset=['Age'], inplace=True)
clinical_df['Gender'] = clinical_df['Gender'].str.upper().map({'M': 'Male', 'F': 'Female'})

print("\nCleaned Data (Sample):")
print(clinical_df.head(), "\n")

# ====================================================
# 3) LOAD MEDICAL IMAGES
# ====================================================

print("Loading medical images from Dataset_BUSI_with_GT...")

image_folder = "Dataset_BUSI_with_GT"
image_entries = glob.glob(f"{image_folder}/**/*.png", recursive=True)
image_entries += glob.glob(f"{image_folder}/**/*.jpg", recursive=True)
image_entries += glob.glob(f"{image_folder}/**/*.jpeg", recursive=True)

print(f"Found total {len(image_entries)} image files.\n")

if len(image_entries) > 0:
    for img_path in image_entries[:3]:
        try:
            img = Image.open(img_path).convert("L").resize((224, 224))
            plt.imshow(img, cmap='gray')
            plt.title(f"Preview: {os.path.basename(img_path)}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not open {img_path}: {e}")
else:
    print("No image files found! Please check dataset path.\n")

# ====================================================
# 4) ICD-10 CODE MAPPING
# ====================================================

icd10_map = {
    'C50.9': 'Malignant neoplasm of breast, unspecified',
    'D05.1': 'Lobular carcinoma in situ (benign)',
    'C50.3': 'Malignant neoplasm of upper-inner quadrant of breast',
    'N63': 'Unspecified lump in breast (normal)'
}

def match_image_by_diagnosis(icd_code, image_list):
    """Match image based on diagnosis keyword."""
    icd_code = icd_code.upper()
    if "C50" in icd_code:
        for path in image_list:
            if "malignant" in path.lower():
                return path
    elif "D05" in icd_code:
        for path in image_list:
            if "benign" in path.lower():
                return path
    elif "N63" in icd_code:
        for path in image_list:
            if "normal" in path.lower():
                return path
    return None

# ====================================================
# 5) CREATE TRAINING DATASETS
# ====================================================

genai_jsonl_data = []
vlm_csv_data = []

for _, row in clinical_df.iterrows():
    pid = row['Patient_ID']
    age, gender = int(row['Age']), row['Gender']
    code = row['ICD10_Code']
    diagnosis = icd10_map.get(code, code)
    notes = row['Doctor_Notes']

    matched_img = match_image_by_diagnosis(code, image_entries)

    if matched_img:
        prompt = (
            f"Patient ({gender}, {age} years old):\n"
            f"Doctor Observation: {notes}\n"
            f"What is the likely diagnosis?"
        )
        genai_jsonl_data.append({
            "patient_id": pid,
            "image_file": matched_img,
            "prompt": prompt,
            "response": diagnosis
        })
        vlm_csv_data.append({
            "patient_id": pid,
            "image_file": matched_img,
            "text_prompt": notes,
            "label": diagnosis
        })
    else:
        print(f"No image found for {pid} ({code})")

with open("genai_dataset.jsonl", "w", encoding='utf-8') as f:
    for entry in genai_jsonl_data:
        f.write(json.dumps(entry) + '\n')

pd.DataFrame(vlm_csv_data).to_csv("vlm_dataset.csv", index=False)

print("\nGenerated GenAI + VLM training datasets successfully.\n")

# ====================================================
# 6) VISUAL ANALYSIS
# ====================================================

plt.figure(figsize=(8, 5))
clinical_df['ICD10_Code'].map(icd10_map).value_counts().plot(
    kind='bar', color='skyblue', edgecolor='black'
)
plt.title("Diagnosis Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(clinical_df['Age'], bins=10, color='lightgreen', edgecolor='black')
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ====================================================
# 7) CLINICAL NOTE GENERATION + ICD PREDICTION
# ====================================================

print("\nGenerating clinical notes & predicting ICD-10 codes...\n")

def predict_icd10(text):
    t = text.lower()
    if "lump" in t or "mass" in t: return "N63"
    if "carcinoma" in t or "malignant" in t: return "C50.9"
    if "benign" in t or "in situ" in t: return "D05.1"
    return "C50.3"

def generate_clinical_note(row):
    return (f"{row['Age']}-year-old {row['Gender']} patient shows clinical signs of "
            f"{icd10_map.get(row['ICD10_Code'], 'breast abnormality')}. "
            f"Observation summary: {row['Doctor_Notes']}")

clinical_df["Generated_Notes"] = clinical_df.apply(generate_clinical_note, axis=1)
clinical_df["Predicted_ICD_Code"] = clinical_df["Generated_Notes"].apply(predict_icd10)
clinical_df["Predicted_ICD_Description"] = clinical_df["Predicted_ICD_Code"].map(icd10_map)

# Display sample output
print("Sample Output:\n")
print("{:<10} {:<80} {:<20} {:<50}".format(
    "Patient_ID", "Generated_Notes", "Predicted_ICD_Code", "Predicted_ICD_Description"
))
print("-" * 170)
for _, row in clinical_df.iterrows():
    print("{:<10} {:<80} {:<20} {:<50}".format(
        row["Patient_ID"],
        row["Generated_Notes"][:75] + "...",
        row["Predicted_ICD_Code"],
        row["Predicted_ICD_Description"]
    ))

clinical_df.to_csv("final_clinical_output.csv", index=False)
print("\nFinal AI-Generated output saved as: final_clinical_output.csv")
print("Module completed successfully!")

# ====================================================
# 8) OPTIONAL: UNSTRUCTURED DATA ONLY
# ====================================================

ask_extra = input("\nDo you also want to generate from *unstructured text only*? (yes/no): ").strip().lower()

if ask_extra == "yes":
    print("\nGenerating unstructured clinical notes...\n")
    unstructured_df["Generated_Note"] = unstructured_df["Doctor_Notes"].apply(
        lambda x: f"Unstructured clinical summary: {x}"
    )
    unstructured_df.to_csv("unstructured_generated_notes.csv", index=False)
    print("Saved unstructured-only notes as unstructured_generated_notes.csv\n")
else:
    print("\nSkipped unstructured-only generation.\n")


