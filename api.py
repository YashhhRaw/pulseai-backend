from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PulseAI ML Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("lgbm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

class PatientInput(BaseModel):
    gender: int
    age: int
    history: int
    medication: int
    severity: int
    breath: int
    vision: int
    nose: int
    systolic: float
    diastolic: float
    diet: int

@app.get("/")
def root():
    return {"status": "PulseAI backend running"}

@app.post("/predict")
def predict(data: PatientInput):
    X = [[
        data.gender,
        data.age,
        data.history,
        data.medication,
        data.severity,
        data.breath,
        data.vision,
        data.nose,
        data.systolic,
        data.diastolic,
        data.diet
    ]]

    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    prob = float(max(model.predict_proba(X_scaled)[0]) * 100)

    labels = [
        "Normal",
        "Stage-1 Hypertension",
        "Stage-2 Hypertension",
        "Hypertensive Crisis"
    ]

    return {
        "stage": labels[pred],
        "confidence": round(prob, 1)
    }
