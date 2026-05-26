import requests
import time
from datetime import datetime, timedelta
from typing import Dict

COORD_AEROPORT = {
    "LFPG": (49.0097, 2.5479),
    "LFPO": (48.7253, 2.3594),
    "LFMN": (43.6653, 7.2150),
    "LFLL": (45.7264, 5.0908),
    "LFML": (43.4393, 5.2214),
}

HOURLY_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_gusts_10m,pressure_msl,precipitation,cloud_cover"


def _get_meteo(lat: float, lon: float, start: str, end: str, is_future: bool) -> Dict:
    base = "https://api.open-meteo.com/v1/forecast" if is_future else "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": HOURLY_VARS,
        "timezone": "Europe/Paris"
    }
    try:
        r = requests.get(base, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Erreur meteo {start}→{end} : {e}")
        return {}


def _fetch_meteo(start_date: datetime.date, end_date: datetime.date, is_future: bool = False) -> Dict[str, Dict]:
    """Récupère la météo brute pour chaque aéroport"""
    data = {}
    for icao, (lat, lon) in COORD_AEROPORT.items():
        meteo = _get_meteo(lat, lon, start_date.isoformat(), end_date.isoformat(), is_future)
        data[f"{icao}_raw"] = meteo
        time.sleep(0.7)
    return data
