import argparse
import pandas as pd
import io
from custom_libs.load_from import load_from_s3
from custom_libs.save_to import save_to_s3


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputation des valeurs manquantes pour les colonnes critiques."""
    df = df.copy()
    
    # Imputation simple et sûre
    df["terminal"] = df["terminal"].fillna("Unknown")
    df["aircraft_model"] = df["aircraft_model"].fillna("Unknown")
    #df["airline_icao"] = df["airline_icao"].fillna("Unknown")
    
    remaining = df.isnull().sum().sum()
    print(f"   - Valeurs manquantes restantes après imputation : {remaining}")
    
    return df


def impute_and_save(run_id: str, is_future: bool = False):
    mode = "FUTUR" if is_future else "HISTORIQUE"
    print(f"=== Imputation {mode} - Run ID: {run_id} ===\n")
    
    folder = "prediction" if is_future else "train"
    prefix = "clean"
    
    # === Arrivées ===
    print("→ Imputation Arrivées...")
    arrive_bytes = load_from_s3(f"processed/{folder}/{run_id}", f"{prefix}_arrivals_{run_id}.parquet")
    df_arrive = pd.read_parquet(io.BytesIO(arrive_bytes))
    
    df_arrive = impute_missing_values(df_arrive)
    
    buffer = io.BytesIO()
    df_arrive.to_parquet(buffer, index=False, compression="gzip")
    save_to_s3(buffer.getvalue(), f"processed/{folder}/{run_id}", f"imputed_arrivals_{run_id}.parquet")
    print(f"   ✓ Arrivées imputées : {len(df_arrive)} lignes\n")
    
    # === Départs ===
    print("→ Imputation Départs...")
    depart_bytes = load_from_s3(f"processed/{folder}/{run_id}", f"{prefix}_departures_{run_id}.parquet")
    df_depart = pd.read_parquet(io.BytesIO(depart_bytes))
    
    df_depart = impute_missing_values(df_depart)
    
    buffer = io.BytesIO()
    df_depart.to_parquet(buffer, index=False, compression="gzip")
    save_to_s3(buffer.getvalue(), f"processed/{folder}/{run_id}", f"imputed_departures_{run_id}.parquet")
    print(f"   ✓ Départs imputés : {len(df_depart)} lignes\n")
    
    print("=== Imputation terminée ===\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--is_future", action="store_true")
    args = parser.parse_args()
    
    impute_and_save(args.run_id, args.is_future)


if __name__ == "__main__":
    main()