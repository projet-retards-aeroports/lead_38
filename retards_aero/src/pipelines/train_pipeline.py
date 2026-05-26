import argparse
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import mlflow
import mlflow.catboost
import os
from dotenv import load_dotenv

from custom_libs.load_from import load_from_s3

load_dotenv()

# ====================== CONFIGURATION MLFLOW ======================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("projet-retards-avion")

print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}\n")


def prepare_data(df: pd.DataFrame) -> tuple:
    df = df.copy()
    
    # Conversion datetime
    df["scheduled_utc"] = pd.to_datetime(df["scheduled_utc"], errors="coerce", utc=True)
    df["revised_utc"] = pd.to_datetime(df["revised_utc"], errors="coerce", utc=True)
    
    # Cible
    df["delay_minutes"] = (df["revised_utc"] - df["scheduled_utc"]).dt.total_seconds() / 60
    
    return df


def add_avg_delay_feature(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Ajoute avg_delay_per_flight calculé uniquement sur le train"""
    df = df.copy()
    
    if is_train:
        # Calcul sur tout le dataset actuel (pour l'instant)
        avg_delay = df.groupby("flight_number")["delay_minutes"].mean().reset_index()
        avg_delay.columns = ["flight_number", "avg_delay_per_flight"]
        df = df.merge(avg_delay, on="flight_number", how="left")

        # === DEBUG : Affichage des exemples ===
        print(f"\nExemples de avg_delay_per_flight ({len(avg_delay)} vols uniques) :")
        sample = avg_delay.sample(n=min(10, len(avg_delay)), random_state=42)
        print(sample.round(2).to_string(index=False))
        
        # Statistiques globales
        print(f"Moyenne globale du délai moyen : {avg_delay['avg_delay_per_flight'].mean():.2f} minutes")
        print(f"Min / Max délai moyen         : {avg_delay['avg_delay_per_flight'].min():.2f} / {avg_delay['avg_delay_per_flight'].max():.2f} minutes")
    else:
        # Pour l'inférence future, on utilisera une table pré-calculée
        pass
    
    return df


def train_model(df, model_name: str, run_id: str):
    with mlflow.start_run(run_name=f"{model_name}_{run_id}") as run:
        
        print(f"\n=== Entraînement {model_name} ===")
        
        df = prepare_data(df)
        df = add_avg_delay_feature(df, is_train=True)
        
        # Features + Cible
        X = df.drop(columns=["scheduled_utc", "revised_utc", "flight_number", "delay_minutes"])
        y = df["delay_minutes"]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
        cat_features = [
            "aeroport_depart", "aeroport_arrivee", "terminal", "airline_icao",
            "aircraft_model", "aircraft_family", "aircraft_size_category", 
            "holiday_name"
        ]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            early_stopping_rounds=300,
            verbose=50,
            task_type='CPU',
            l2_leaf_reg=3,
            random_strength=1.0,
            bagging_temperature=0.7
        )
        
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        # Évaluation
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        
        mlflow.log_metric("MAE", round(mae, 4))
        mlflow.log_metric("RMSE", round(rmse, 4))
        mlflow.log_metric("R2", round(r2, 4))
        mlflow.log_metric("best_iteration", model.get_best_iteration())
        
        print(f"\n=== Résultats {model_name} ===")
        print(f"MAE   : {mae:.3f} minutes")
        print(f"RMSE  : {rmse:.3f} minutes")
        print(f"R²    : {r2:.4f}")
        print(f"Best iteration : {model.get_best_iteration()}")
        print("="*60 + "\n")


def train_pipeline(run_id: str = None):
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    print(f"=== Lancement du Train Pipeline - Run ID: {run_id} ===\n")
    
    # Départs
    depart_bytes = load_from_s3(f"processed/train/{run_id}", f"final_departures_{run_id}.parquet")
    df_depart = pd.read_parquet(io.BytesIO(depart_bytes))
    train_model(df_depart, "Departure", run_id)
    
    # Arrivées
    arrive_bytes = load_from_s3(f"processed/train/{run_id}", f"final_arrivals_{run_id}.parquet")
    df_arrive = pd.read_parquet(io.BytesIO(arrive_bytes))
    train_model(df_arrive, "Arrival", run_id)
    
    print(f"\n=== Train Pipeline terminé ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()
    train_pipeline(args.run_id)


if __name__ == "__main__":
    main()