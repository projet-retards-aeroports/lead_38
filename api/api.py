from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import mlflow
import os
from typing import List, Dict, Any

# ====================== APP CONFIG ======================
app = FastAPI(
    title="✈️ Retards Aéro - API de Prédiction",
    description="""
    API de prédiction des retards de vols aériens.
    ### Fonctionnalités :
    - Prédiction des retards au départ (`departures`)
    - Prédiction des retards à l'arrivée (`arrivals`)
    - Modèles CatBoost entraînés avec MLflow
    ### Comment utiliser :
    1. Envoyer une liste de vols
    2. Préciser le type de vol (`departures` ou `arrivals`)
    3. Récupérer les prédictions en minutes
    """,
    version="1.0.0",
    contact={
        "name": "Équipe Projet Lead",
        "url": "https://huggingface.co/spaces/projetLead38/pl_mlflow",
    },
    license_info={
        "name": "Ludovic et Patrick FullStack 40 & Lead 38",
    },
)

# ====================== MLFLOW ======================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
print(" Chargement des modèles Production...")

model_depart = None
model_arrive = None

try:
    model_depart = mlflow.pyfunc.load_model("models:/CatBoost_Departure_Delay_Prediction@production")
    model_arrive = mlflow.pyfunc.load_model("models:/CatBoost_Arrival_Delay_Prediction@production")
    print(" Les 2 modèles sont chargés avec succès (Production)")
except Exception as e:
    print(f" Erreur chargement modèles: {e}")
    raise

# ====================== SCHEMAS ======================
class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Liste des vols à prédire")
    flight_type: str = Field(..., description="Type de vol : 'departures' ou 'arrivals'")
    run_id: str = Field(..., description="Identifiant du run MLflow (fourni par Streamlit)")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "flight_number": "AF1234",
                        "scheduled_utc": "2026-05-28T14:30:00",
                        "aeroport_depart": "CDG",
                        "aeroport_arrivee": "JFK",
                        "airline_name": "Air France"
                    }
                ],
                "flight_type": "departures",
                "run_id": "2026-05-25_221052_8f2d98"
            }
        }

class PredictionResponse(BaseModel):
    status: str
    run_id: str
    flight_type: str
    predictions: List[float]
    count: int

# ====================== ROUTES ======================
@app.get("/", tags=["Health"])
def root():
    return {"message": "API Retards Aéro - OK", "status": "running"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prédictions"])
def predict(request: PredictionRequest):
    if request.flight_type not in ["departures", "arrivals"]:
        raise HTTPException(status_code=400, detail="flight_type invalide (doit être 'departures' ou 'arrivals')")

    model = model_depart if request.flight_type == "departures" else model_arrive

    try:
        df = pd.DataFrame(request.data)
        features = df.drop(columns=["scheduled_utc", "flight_number"], errors="ignore")
        predictions = model.predict(features)

        return {
            "status": "success",
            "run_id": request.run_id,
            "flight_type": request.flight_type,
            "predictions": [round(float(p), 2) for p in predictions],
            "count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# ====================== LANCEMENT ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
