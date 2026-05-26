import argparse
from datetime import datetime, timedelta
import json
import io

from src.fetcher_vols import _fetch_vols
from src.fetcher_meteo import _fetch_meteo
from src.merge_vols_meteo import merge_vols_meteo, merged_to_departures_dataframe, merged_to_arrivals_dataframe
from src.save_to import save_to_s3
from src.load_from import load_from_s3

import uuid

def run_full_pipeline():
    start = datetime.now()
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    print(f"=== Pipeline complet - Run ID: {run_id} ===\n")

    today = datetime.now().date()

    # === 1. Vols ===
    print("1/5 - Récupération des vols (180 jours)...")
    vols_data = _fetch_vols(today - timedelta(days=180), today - timedelta(days=1))
    save_to_s3(json.dumps(vols_data, default=str).encode(), f"raw/merged_meteo_vols/{run_id}", f"vols_{run_id}.json")
    print(f"   → {len(vols_data)} entrées sauvegardées\n")
    del vols_data

    # === 2. Météo ===
    print("2/5 - Récupération de la météo (181 jours)...")
    meteo_data = _fetch_meteo(today - timedelta(days=181), today - timedelta(days=1))
    save_to_s3(json.dumps(meteo_data, default=str).encode(), f"raw/merged_meteo_vols/{run_id}", f"meteo_{run_id}.json")
    print(f"   → {len(meteo_data)} entrées sauvegardées\n")
    del meteo_data

    # === 3. Merge (on recharge seulement ce dont on a besoin) ===
    print("3/5 - Fusion...")
    merged = merge_vols_meteo(
        json.loads(load_from_s3(f"raw/merged_meteo_vols/{run_id}", f"vols_{run_id}.json")),
        json.loads(load_from_s3(f"raw/merged_meteo_vols/{run_id}", f"meteo_{run_id}.json"))
    )
    save_to_s3(json.dumps(merged, default=str).encode(), f"raw/merged_meteo_vols/{run_id}", f"merged_{run_id}.json")
    print(f"   → Fusion terminée\n")

    # === 4. Dataset Départs ===
    print("4/5 - Création du dataset départs...")
    df_depart = merged_to_departures_dataframe(merged)
    buffer = io.BytesIO()
    df_depart.to_parquet(buffer, index=False, compression="gzip")
    save_to_s3(buffer.getvalue(), f"processed/train/{run_id}", f"departures_{run_id}.parquet")
    print(f"   → {len(df_depart)} lignes\n")
    del df_depart, buffer

    # === 5. Dataset Arrivées ===
    print("5/5 - Création du dataset arrivées...")
    df_arrive = merged_to_arrivals_dataframe(merged)
    buffer = io.BytesIO()
    df_arrive.to_parquet(buffer, index=False, compression="gzip")
    save_to_s3(buffer.getvalue(), f"processed/train/{run_id}", f"arrivals_{run_id}.parquet")
    print(f"   → {len(df_arrive)} lignes\n")
    del df_arrive, buffer, merged   # ← Nettoyage final
    print(f"=== Pipeline terminé en {round((datetime.now() - start).total_seconds(), 1)} secondes ===")


def main():
    parser = argparse.ArgumentParser(description="Pipeline complet vols + météo + datasets")
    parser.add_argument("--run_id", type=str, default=None, help="Optionnel : forcer un run_id")
    args = parser.parse_args()

    if args.run_id:
        print(f"Run ID forcé : {args.run_id}")
        # Tu peux ajouter la logique si tu veux supporter un run_id forcé
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()