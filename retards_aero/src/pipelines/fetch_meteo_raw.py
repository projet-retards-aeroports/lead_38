import argparse
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pipelines.custom_libs.fetcher_meteo import _fetch_meteo
from pipelines.custom_libs.save_to import save_to_s3

load_dotenv()

def fetch_meteo_raw(run_id: str, is_future: bool = False):
    print(f"=== Récupération de la météo {'FUTURE' if is_future else 'HISTORIQUE'} ===\n")
    
    start = datetime.now()
    today = datetime.now().date()
    
    if is_future:
        start_meteo = today + timedelta(days=0)   # J
        end_meteo = today + timedelta(days=8)     # J+8
        print(f"Période météo future : J à J+8 ({start_meteo} → {end_meteo})")
    else:
        start_meteo = today - timedelta(days=181)
        end_meteo = today - timedelta(days=1)
        print(f"Période météo historique : J-181 à J-1 ({start_meteo} → {end_meteo})")

    meteo_data = _fetch_meteo(start_meteo, end_meteo,is_future=is_future)

    meteo_bytes = json.dumps(meteo_data, ensure_ascii=False, default=str).encode("utf-8")
    save_to_s3(meteo_bytes, f"raw/merged_meteo_vols/{run_id}", f"meteo_{run_id}.json")

    duration = (datetime.now() - start).total_seconds()

    print(f"Terminé en {round(duration, 2)} secondes | {len(meteo_data)} entrées sauvegardées")
    print(f"✓ Météo {'future' if is_future else 'historique'} sauvegardée\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--is_future", action="store_true")
    args = parser.parse_args()
    
    fetch_meteo_raw(args.run_id, args.is_future)


if __name__ == "__main__":
    main()