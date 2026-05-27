from datetime import datetime, timedelta
from typing import Dict
import pandas as pd
from tqdm import tqdm


def merge_vols_meteo(vols_data: dict, raw_meteo_data: dict, show_progress: bool = True) -> dict:
    """Fusion vols + météo avec barre de progression"""
    merged = {}
    stats = {"departs": 0, "arrivees": 0}

    items = vols_data.items()
    if show_progress:
        items = tqdm(items, desc="Merge vols + meteo", unit="jour")

    for key, movements in items:
        icao_fr = key.split("_")[0]
        merged[key] = {"departures": [], "arrivals": []}

        meteo_raw = raw_meteo_data.get(f"{icao_fr}_raw", {})
        hourly = meteo_raw.get("hourly", {})
        times = hourly.get("time", [])

        # === DÉPARTS ===
        for flight in movements.get("departures", []):
            flight_copy = flight.copy()
            dep_utc = flight.get("departure", {}).get("scheduledTime", {}).get("utc")
            if dep_utc and times:
                try:
                    target = (datetime.fromisoformat(dep_utc.replace("Z", "+00:00")) - timedelta(hours=3)).replace(tzinfo=None)
                    best_idx = min(range(len(times)), key=lambda i: abs(
                        datetime.fromisoformat(times[i].replace("Z", "+00:00")).replace(tzinfo=None) - target))
                    flight_copy["meteo_3h_avant_depart"] = {k: hourly[k][best_idx] for k in hourly if k != "time"}
                    stats["departs"] += 1
                except:
                    flight_copy["meteo_3h_avant_depart"] = None
            merged[key]["departures"].append(flight_copy)

        # === ARRIVÉES ===
        for flight in movements.get("arrivals", []):
            flight_copy = flight.copy()
            arr_utc = flight.get("arrival", {}).get("scheduledTime", {}).get("utc")
            if arr_utc and times:
                try:
                    target = (datetime.fromisoformat(arr_utc.replace("Z", "+00:00")) - timedelta(hours=3)).replace(tzinfo=None)
                    best_idx = min(range(len(times)), key=lambda i: abs(
                        datetime.fromisoformat(times[i].replace("Z", "+00:00")).replace(tzinfo=None) - target))
                    flight_copy["meteo_3h_avant_arrivee"] = {k: hourly[k][best_idx] for k in hourly if k != "time"}
                    stats["arrivees"] += 1
                except:
                    flight_copy["meteo_3h_avant_arrivee"] = None
            merged[key]["arrivals"].append(flight_copy)

    print(f"\n=== STATS MERGE ===")
    print(f"Départs  : {stats['departs']}")
    print(f"Arrivées : {stats['arrivees']}")
    return merged


# ====================== FONCTION JSON → DATAFRAME ======================

def merged_to_dataframe(merged: dict, show_progress: bool = True) -> pd.DataFrame:
    """Convertit le merged en DataFrame complet"""
    rows = []
    items = merged.items()
    if show_progress:
        items = tqdm(items, desc="Conversion DataFrame complet", unit="jour")

    for key, movements in items:
        icao_fr = key.split("_")[0]
        day = key.split("_")[1]

        for flight in movements.get("departures", []):
            row = {
                "key": key, "aeroport_depart": icao_fr, "aeroport_arrivee": flight.get("arrival", {}).get("airport", {}).get("icao"),
                "date": day, "type": "departure",
                "dep_scheduled_utc": flight.get("departure", {}).get("scheduledTime", {}).get("utc"),
                "dep_scheduled_local": flight.get("departure", {}).get("scheduledTime", {}).get("local"),
                "dep_revised_utc": flight.get("departure", {}).get("revisedTime", {}).get("utc"),
                "dep_revised_local": flight.get("departure", {}).get("revisedTime", {}).get("local"),
                "dep_runway_utc": flight.get("departure", {}).get("runwayTime", {}).get("utc"),
                "dep_terminal": flight.get("departure", {}).get("terminal"),
                "dep_gate": flight.get("departure", {}).get("gate"),
                "arr_airport_icao": flight.get("arrival", {}).get("airport", {}).get("icao"),
                "arr_airport_iata": flight.get("arrival", {}).get("airport", {}).get("iata"),
                "arr_airport_name": flight.get("arrival", {}).get("airport", {}).get("name"),
                "arr_scheduled_utc": flight.get("arrival", {}).get("scheduledTime", {}).get("utc"),
                "arr_scheduled_local": flight.get("arrival", {}).get("scheduledTime", {}).get("local"),
                "arr_revised_utc": flight.get("arrival", {}).get("revisedTime", {}).get("utc"),
                "arr_revised_local": flight.get("arrival", {}).get("revisedTime", {}).get("local"),
                "arr_runway_utc": flight.get("arrival", {}).get("runwayTime", {}).get("utc"),
                "arr_terminal": flight.get("arrival", {}).get("terminal"),
                "arr_gate": flight.get("arrival", {}).get("gate"),
                "arr_baggage_belt": flight.get("arrival", {}).get("baggageBelt"),
                "flight_number": flight.get("number"),
                "callsign": flight.get("callSign"),
                "status": flight.get("status"),
                "codeshare_status": flight.get("codeshareStatus"),
                "is_cargo": flight.get("isCargo"),
                "aircraft_reg": flight.get("aircraft", {}).get("reg"),
                "aircraft_model": flight.get("aircraft", {}).get("model"),
                "aircraft_modeS": flight.get("aircraft", {}).get("modeS"),
                "airline_name": flight.get("airline", {}).get("name"),
                "airline_iata": flight.get("airline", {}).get("iata"),
                "airline_icao": flight.get("airline", {}).get("icao"),
                **{f"meteo_{k}": v for k, v in flight.get("meteo_3h_avant_depart", {}).items()}
            }
            rows.append(row)

        for flight in movements.get("arrivals", []):
            row = {
                "key": key, "aeroport_depart": flight.get("departure", {}).get("airport", {}).get("icao"),
                "aeroport_arrivee": icao_fr, "date": day, "type": "arrival",
                "dep_airport_icao": flight.get("departure", {}).get("airport", {}).get("icao"),
                "dep_airport_iata": flight.get("departure", {}).get("airport", {}).get("iata"),
                "dep_airport_name": flight.get("departure", {}).get("airport", {}).get("name"),
                "dep_scheduled_utc": flight.get("departure", {}).get("scheduledTime", {}).get("utc"),
                "dep_scheduled_local": flight.get("departure", {}).get("scheduledTime", {}).get("local"),
                "dep_revised_utc": flight.get("departure", {}).get("revisedTime", {}).get("utc"),
                "dep_revised_local": flight.get("departure", {}).get("revisedTime", {}).get("local"),
                "dep_runway_utc": flight.get("departure", {}).get("runwayTime", {}).get("utc"),
                "dep_terminal": flight.get("departure", {}).get("terminal"),
                "dep_gate": flight.get("departure", {}).get("gate"),
                "dep_checkin_desk": flight.get("departure", {}).get("checkInDesk"),
                "arr_scheduled_utc": flight.get("arrival", {}).get("scheduledTime", {}).get("utc"),
                "arr_scheduled_local": flight.get("arrival", {}).get("scheduledTime", {}).get("local"),
                "arr_revised_utc": flight.get("arrival", {}).get("revisedTime", {}).get("utc"),
                "arr_revised_local": flight.get("arrival", {}).get("revisedTime", {}).get("local"),
                "arr_runway_utc": flight.get("arrival", {}).get("runwayTime", {}).get("utc"),
                "arr_terminal": flight.get("arrival", {}).get("terminal"),
                "arr_gate": flight.get("arrival", {}).get("gate"),
                "arr_baggage_belt": flight.get("arrival", {}).get("baggageBelt"),
                "arr_runway": flight.get("arrival", {}).get("runway"),
                "flight_number": flight.get("number"),
                "callsign": flight.get("callSign"),
                "status": flight.get("status"),
                "codeshare_status": flight.get("codeshareStatus"),
                "is_cargo": flight.get("isCargo"),
                "aircraft_reg": flight.get("aircraft", {}).get("reg"),
                "aircraft_model": flight.get("aircraft", {}).get("model"),
                "aircraft_modeS": flight.get("aircraft", {}).get("modeS"),
                "airline_name": flight.get("airline", {}).get("name"),
                "airline_iata": flight.get("airline", {}).get("iata"),
                "airline_icao": flight.get("airline", {}).get("icao"),
                **{f"meteo_{k}": v for k, v in flight.get("meteo_3h_avant_arrivee", {}).items()}
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ====================== SPLIT DÉPARTS / ARRIVÉES ======================

def merged_to_departures_dataframe(merged: dict, show_progress: bool = True) -> pd.DataFrame:
    """Dataset départs uniquement"""
    rows = []
    items = merged.items()
    if show_progress:
        items = tqdm(items, desc="Dataset Départs", unit="jour")

    for key, movements in items:
        icao_fr = key.split("_")[0]
        day = key.split("_")[1]

        for flight in movements.get("departures", []):
            row = {
                "key": key,
                "aeroport_depart": icao_fr,
                "aeroport_arrivee": flight.get("arrival", {}).get("airport", {}).get("icao"),
                "date": day,
                "type": "departure",
                "dep_scheduled_utc": flight.get("departure", {}).get("scheduledTime", {}).get("utc"),
                "dep_revised_utc": flight.get("departure", {}).get("revisedTime", {}).get("utc"),
                "dep_runway_utc": flight.get("departure", {}).get("runwayTime", {}).get("utc"),
                "dep_terminal": flight.get("departure", {}).get("terminal"),
                "flight_number": flight.get("number"),
                "callsign": flight.get("callSign"),
                "status": flight.get("status"),
                "codeshare_status": flight.get("codeshareStatus"),
                "is_cargo": flight.get("isCargo"),
                "aircraft_model": flight.get("aircraft", {}).get("model"),
                "airline_icao": flight.get("airline", {}).get("icao"),
                "airline_name": flight.get("airline", {}).get("name"),
                **{f"meteo_{k}": v for k, v in flight.get("meteo_3h_avant_depart", {}).items() if flight.get("meteo_3h_avant_depart")}
            }
            rows.append(row)

    return pd.DataFrame(rows)


def merged_to_arrivals_dataframe(merged: dict, show_progress: bool = True) -> pd.DataFrame:
    """Dataset arrivées uniquement"""
    rows = []
    items = merged.items()
    if show_progress:
        items = tqdm(items, desc="Dataset Arrivées", unit="jour")

    for key, movements in items:
        icao_fr = key.split("_")[0]
        day = key.split("_")[1]

        for flight in movements.get("arrivals", []):
            row = {
                "key": key,
                "aeroport_depart": flight.get("departure", {}).get("airport", {}).get("icao"),
                "aeroport_arrivee": icao_fr,
                "date": day,
                "type": "arrival",
                "arr_scheduled_utc": flight.get("arrival", {}).get("scheduledTime", {}).get("utc"),
                "arr_revised_utc": flight.get("arrival", {}).get("revisedTime", {}).get("utc"),
                "arr_runway_utc": flight.get("arrival", {}).get("runwayTime", {}).get("utc"),
                "arr_terminal": flight.get("arrival", {}).get("terminal"),
                "flight_number": flight.get("number"),
                "callsign": flight.get("callSign"),
                "status": flight.get("status"),
                "codeshare_status": flight.get("codeshareStatus"),
                "is_cargo": flight.get("isCargo"),
                "aircraft_model": flight.get("aircraft", {}).get("model"),
                "airline_icao": flight.get("airline", {}).get("icao"),
                "airline_name": flight.get("airline", {}).get("name"),
                **{f"meteo_{k}": v for k, v in flight.get("meteo_3h_avant_arrivee", {}).items() if flight.get("meteo_3h_avant_arrivee")}
            }
            rows.append(row)

    return pd.DataFrame(rows)