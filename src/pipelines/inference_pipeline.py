import pandas as pd
import joblib
from typing import Dict, Any, List
import io

from src.load_from import load_from_s3


def load_latest_model(model_type: str):
    """Charge le dernier modèle entraîné (à améliorer plus tard avec MLflow)"""
    # Pour l’instant on charge le dernier run_id disponible
    # Plus tard on utilisera MLflow pour charger le modèle "Champion"
    if model_type == "departure":
        model_path = "models/departure_model_latest.pkl"
    else:
        model_path = "models/arrival_model_latest.pkl"
    
    return joblib.load(model_path)


def predict_single_flight(flight_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prédit le retard d'un seul vol.
    flight_data doit contenir :
    - type: "departure" ou "arrival"
    - toutes les features nécessaires
    """
    model_type = flight_data.get("type")
    
    if model_type not in ["departure", "arrival"]:
        return {"status": "error", "message": "Type doit être 'departure' ou 'arrival'"}
    
    try:
        model = load_latest_model(model_type)
        df = pd.DataFrame([flight_data])
        prediction = model.predict(df)[0]
        
        return {
            "status": "success",
            "type": model_type,
            "predicted_delay_minutes": round(float(prediction), 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def predict_multiple_flights(flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prédit plusieurs vols (batch)"""
    results = []
    for flight in flights:
        result = predict_single_flight(flight)
        results.append(result)
    return results


# ====================== Test local ======================
if __name__ == "__main__":
    sample = {
        "type": "departure",
        "dep_scheduled_utc": "2026-05-26T10:00:00",
        "aircraft_model": "Airbus A320",
        "airline_icao": "AFR",
        # ... autres features
    }
    
    print(predict_single_flight(sample))