import argparse
import json
import io
from datetime import datetime
from save_to import save_to_s3
from load_from import load_from_s3
from merge_vols_meteo import merged_to_arrivals_dataframe


def create_datasets_arrivals(run_id: str,is_future: bool = False):
    """Génération du dataset arrivées + sauvegarde Parquet sur S3"""
    start = datetime.now()

    print(f"Création du dataset arrivées pour le run_id : {run_id}")

    # Chargement du merged
    merged_bytes = load_from_s3(f"raw/merged_meteo_vols/{run_id}", f"merged_vols_meteo_{run_id}.json")
    merged = json.loads(merged_bytes)

    # Génération du DataFrame arrivées
    df_arrive = merged_to_arrivals_dataframe(merged)

    # Sauvegarde en Parquet
    buffer = io.BytesIO()
    df_arrive.to_parquet(buffer, index=False, compression="gzip")
    if is_future:
        save_to_s3(buffer.getvalue(), f"processed/prediction/{run_id}", f"base_dataset_prediction_arrivals_{run_id}.parquet")
    else:
        save_to_s3(buffer.getvalue(), f"processed/train/{run_id}", f"base_dataset_train_arrivals_{run_id}.parquet")

    duration = (datetime.now() - start).total_seconds()

    print(f"Terminé en {round(duration, 2)} secondes | {len(df_arrive)} lignes sauvegardées")


def main():
    parser = argparse.ArgumentParser(description="Création du dataset arrivées")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID (ex: 2026-05-25_100442)")
    parser.add_argument("--is_future", action="store_true")  
    args = parser.parse_args()

    create_datasets_arrivals(args.run_id, args.is_future)


if __name__ == "__main__":
    main()