import argparse
import pandas as pd
import io
from src.load_from import load_from_s3
from src.save_to import save_to_s3


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les préfixes arr_ et dep_"""
    df = df.copy()
    new_columns = {}
    for col in df.columns:
        if col.startswith("arr_"):
            new_columns[col] = col.replace("arr_", "", 1)
        elif col.startswith("dep_"):
            new_columns[col] = col.replace("dep_", "", 1)
        else:
            new_columns[col] = col
    return df.rename(columns=new_columns)


def clean_dataset(df: pd.DataFrame, is_future: bool = False) -> pd.DataFrame:
    """Nettoyage principal"""
    df = df.copy()
    
    if is_future:
        # === Pour données FUTURES ===
        initial = len(df)
        
        # 1. Garder uniquement les vols "Expected"
        df = df[df["status"] == "Expected"]
        
        # 2. Supprimer les lignes sans aéroport
        df = df.dropna(subset=["aeroport_depart", "aeroport_arrivee"])
        
        print(f"   - Lignes supprimées (status != Expected ou aéroport manquant) : {initial - len(df)}")
        
        # 3. Suppression des colonnes inutiles
        columns_to_drop = [
            "key", "date", "type", "callsign", "status", 
            "codeshare_status", "runway_utc", "revised_utc"
        ]
        
    else:
        # === Pour données HISTORIQUES ===
        initial = len(df)
        df = df.dropna(subset=["scheduled_utc", "revised_utc", "aeroport_arrivee", 
                               "aeroport_depart", "airline_icao"])
        print(f"   - Lignes supprimées (NaN) : {initial - len(df)}")
        
        columns_to_drop = [
            "key", "date", "type", "callsign", "status", 
            "codeshare_status", "runway_utc"
        ]
    
    # Suppression des colonnes
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols)
    
    return df


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip des outliers (uniquement historique)"""
    df = df.copy()
    df["scheduled_utc"] = pd.to_datetime(df["scheduled_utc"], errors="coerce", utc=True)
    df["revised_utc"] = pd.to_datetime(df["revised_utc"], errors="coerce", utc=True)
    
    df["delay_minutes"] = (df["revised_utc"] - df["scheduled_utc"]).dt.total_seconds() / 60
    df["delay_minutes"] = df["delay_minutes"].clip(lower=-120, upper=180)
    
    print(f"   - Délais clippés entre -120 et +180 minutes")
    return df


def clean_data(run_id: str, is_future: bool = False):
    print(f"=== Nettoyage {'FUTUR' if is_future else 'HISTORIQUE'} - Run ID: {run_id} ===\n")
    
    folder = "prediction" if is_future else "train"
    prefix = "base_dataset_prediction" if is_future else "final"
    
    # === Arrivées ===
    print("→ Nettoyage Arrivées...")
    arrive_bytes = load_from_s3(f"processed/{folder}/{run_id}", 
                               f"{prefix}_arrivals_{run_id}.parquet")
    df_arrive = pd.read_parquet(io.BytesIO(arrive_bytes))
    
    df_arrive = standardize_columns(df_arrive)
    df_arrive = clean_dataset(df_arrive, is_future=is_future)
    if not is_future:
        df_arrive = filter_outliers(df_arrive)
    
    buffer = io.BytesIO()
    df_arrive.to_parquet(buffer, index=False, compression="gzip")
    save_to_s3(buffer.getvalue(), f"processed/{folder}/{run_id}", 
               f"clean_arrivals_{run_id}.parquet")
    print(f"   ✓ Arrivées nettoyées : {len(df_arrive)} lignes\n")
    
    # === Départs ===
    print("→ Nettoyage Départs...")
    depart_bytes = load_from_s3(f"processed/{folder}/{run_id}", 
                               f"{prefix}_departures_{run_id}.parquet")
    df_depart = pd.read_parquet(io.BytesIO(depart_bytes))
    
    df_depart = standardize_columns(df_depart)
    df_depart = clean_dataset(df_depart, is_future=is_future)
    if not is_future:
        df_depart = filter_outliers(df_depart)
    
    buffer = io.BytesIO()
    df_depart.to_parquet(buffer, index=False, compression="gzip")
    save_to_s3(buffer.getvalue(), f"processed/{folder}/{run_id}", 
               f"clean_departures_{run_id}.parquet")
    print(f"   ✓ Départs nettoyés : {len(df_depart)} lignes\n")
    
    print("=== Nettoyage terminé ===\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--is_future", action="store_true")
    args = parser.parse_args()
    
    clean_data(args.run_id, args.is_future)


if __name__ == "__main__":
    main()