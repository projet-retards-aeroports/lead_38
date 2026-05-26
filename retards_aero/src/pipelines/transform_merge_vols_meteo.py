import argparse
import json
from datetime import datetime

from load_from import load_from_s3
from save_to import save_to_s3
from merge_vols_meteo import merge_vols_meteo
import os

from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--is_future", action="store_true")  
    args = parser.parse_args()

    run_id = args.run_id
    is_future = args.is_future
    print(f"Execution du merge pour le run_id : {run_id} (FUTURE: {is_future})")

    start = datetime.now()

    # === Chargement + parsing JSON ===
    vols_bytes = load_from_s3(f"raw/merged_meteo_vols/{run_id}", f"vols_{run_id}.json")
    meteo_bytes = load_from_s3(f"raw/merged_meteo_vols/{run_id}", f"meteo_{run_id}.json")

    vols_data = json.loads(vols_bytes)
    meteo_data = json.loads(meteo_bytes)

    print(f"Vols charges : {len(vols_data)} entrees")
    print(f"Meteo chargee : {len(meteo_data)} entrees")

    # === Merge ===
    merged = merge_vols_meteo(vols_data, meteo_data)

    # === Sauvegarde ===
    merged_bytes = json.dumps(merged, ensure_ascii=False, default=str).encode("utf-8")
    save_to_s3(merged_bytes, f"raw/merged_meteo_vols/{run_id}", f"merged_vols_meteo_{run_id}.json")

    duration = (datetime.now() - start).total_seconds()
    print(f"Merge termine en {round(duration, 2)} secondes")


if __name__ == "__main__":
    main()