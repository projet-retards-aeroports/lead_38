import os
from dotenv import load_dotenv
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict
from tqdm import tqdm

load_dotenv()
print("Clé API trouvée :", os.getenv("X_RAPIDAPI_KEY") is not None)

API_HOST = "https://aerodatabox.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": os.getenv("X_RAPIDAPI_KEY"),
    "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com",
}
print (f"API Key Loaded: {os.getenv('X_RAPIDAPI_KEY') is not None}")  # Vérification rapide de la clé
API_PARAMS = {
    "withLeg": "true",
    "direction": "Both",
    "withCancelled": "true",
    "withCodeshared": "true",
    "withCargo": "true",
    "withPrivate": "false",
}
AEROPORTS = {"LFPG": "CDG", "LFPO": "ORY", "LFMN": "NCE", "LFLL": "LYS", "LFML": "MRS"}

def _get_flights(icao: str, from_iso: str, to_iso: str) -> Dict[str, List]:
    """Retourne la structure originale : {'departures': [], 'arrivals': []}"""
    url = f"{API_HOST}/flights/airports/icao/{icao}/{from_iso}/{to_iso}"
    
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=API_PARAMS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            departures = data.get("departures", [])
            arrivals = data.get("arrivals", [])
            
            #print(f" {icao} {from_iso[-5:]} → {len(departures)} dep | {len(arrivals)} arr")
            
            return {
                "departures": departures,
                "arrivals": arrivals
            }
        except Exception as e:
            if getattr(resp, 'status_code', 0) == 429:
                time.sleep(2 ** attempt)
                continue
            print(f"Erreur API {icao} {from_iso}-{to_iso} : {e}")
            return {"departures": [], "arrivals": []}
    return {"departures": [], "arrivals": []}

def _fetch_vols(start_date: datetime.date, end_date: datetime.date) -> Dict[str, Dict]:
    """Structure propre : icao_date → {departures: [], arrivals: []}"""
    data = {}
    current = start_date

    # Barre de progression
    total_days = (end_date - start_date).days + 1
    pbar = tqdm(total=total_days, desc="Récupération vols", unit="jour")
    
    while current <= end_date:
        day_str = current.isoformat()
        for icao in AEROPORTS:
            periods = [
                (f"{day_str}T00:00", f"{day_str}T12:00"),
                (f"{day_str}T12:00", f"{day_str}T23:59")
            ]
            
            all_departs = []
            all_arrivals = []
            
            for fr, to in periods:
                result = _get_flights(icao, fr, to)
                all_departs.extend(result["departures"])
                all_arrivals.extend(result["arrivals"])
                time.sleep(0.8)
            
            key = f"{icao}_{day_str}"
            data[key] = {
                "departures": all_departs,
                "arrivals": all_arrivals
            }
        
        time.sleep(1.2)
        current += timedelta(days=1)
        pbar.update(1)           # ← Mise à jour de la barre
    
    pbar.close()
    return data

