import argparse
import pandas as pd
import io
from src.load_from import load_from_s3


def compare_datasets(run_id_train: str, run_id_pred: str):
    print("=== COMPARAISON COLONNES Train vs Prediction ===\n")
    
    # Chargement
    df_arrive_train = pd.read_parquet(io.BytesIO(
        load_from_s3(f"processed/train/{run_id_train}", f"final_arrivals_{run_id_train}.parquet")
    ))
    df_depart_train = pd.read_parquet(io.BytesIO(
        load_from_s3(f"processed/train/{run_id_train}", f"final_departures_{run_id_train}.parquet")
    ))
    df_arrive_pred = pd.read_parquet(io.BytesIO(
        load_from_s3(f"processed/prediction/{run_id_pred}", f"final_arrivals_{run_id_pred}.parquet")
    ))
    df_depart_pred = pd.read_parquet(io.BytesIO(
        load_from_s3(f"processed/prediction/{run_id_pred}", f"final_departures_{run_id_pred}.parquet")
    ))

    print(f"Train Arrivées  : {len(df_arrive_train.columns)} colonnes")
    print(f"Train Départs   : {len(df_depart_train.columns)} colonnes")
    print(f"Pred Arrivées   : {len(df_arrive_pred.columns)} colonnes")
    print(f"Pred Départs    : {len(df_depart_pred.columns)} colonnes\n")

    common = set(df_arrive_train.columns) & set(df_arrive_pred.columns)
    only_train = set(df_arrive_train.columns) - set(df_arrive_pred.columns)
    only_pred = set(df_arrive_pred.columns) - set(df_arrive_train.columns)

    print("Colonnes communes :", sorted(common))
    print(f"→ Total communes : {len(common)}\n")

    if only_train:
        print("Colonnes UNIQUEMENT dans Train :", sorted(only_train))
    if only_pred:
        print("Colonnes UNIQUEMENT dans Prediction :", sorted(only_pred))


def main():
    parser = argparse.ArgumentParser(description="Comparaison colonnes Train vs Prediction")
    
    parser.add_argument("--train_id", type=str, 
                        default="2026-05-25_124326_039dd1",
                        help="Run ID du Train")
    
    parser.add_argument("--pred_id", type=str, 
                        default="2026-05-25_221052_8f2d98",
                        help="Run ID de la Prediction")
    
    args = parser.parse_args()

    compare_datasets(args.train_id, args.pred_id)


if __name__ == "__main__":
    main()