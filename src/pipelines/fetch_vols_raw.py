import argparse
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

from pipelines.custom_libs.fetcher_vols import _fetch_vols
from pipelines.custom_libs.save_to import save_to_s3

load_dotenv()


def fetch_vols_raw(run_id: str, is_future: bool = False):
    print(f"=== Récupération Vols {'FUTURS' if is_future else 'HISTORIQUES'} ===\n")
    
    today = datetime.now().date()
    
    if is_future:
        start_date = today + timedelta(days=1)
        end_date = today + timedelta(days=8)
        print(f"Période : J+1 → J+8 ({start_date} → {end_date})")
    else:
        start_date = today - timedelta(days=180)
        end_date = today - timedelta(days=1)
        print(f"Période : J-180 → J-1 ({start_date} → {end_date})")
    
    try:
        vols_data = _fetch_vols(start_date, end_date)
        
        vols_bytes = json.dumps(vols_data, ensure_ascii=False, default=str).encode("utf-8")
        save_to_s3(vols_bytes, f"raw/merged_meteo_vols/{run_id}", f"vols_{run_id}.json")
        
        print(f"✓ {len(vols_data)} entrées sauvegardées avec succès")
        
    except Exception as e:
        print(f"Erreur lors de la récupération des vols : {e}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--is_future", action="store_true")
    args = parser.parse_args()
    
    fetch_vols_raw(args.run_id, args.is_future)


if __name__ == "__main__":
    main()