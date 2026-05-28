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

from src.pipelines.custom_libs.load_from import load_from_s3

load_dotenv()

# ====================== CONFIGURATION MLFLOW ======================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("projet-retards-avion")

print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}\n")


def add_avg_delay_feature(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()
    if is_train:
        df = df.sort_values(["flight_number", "scheduled_utc"])
        expanding_avg = df.groupby("flight_number")["delay_minutes"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        flight_mean = df.groupby("flight_number")["delay_minutes"].transform("mean")
        df["avg_delay_per_flight"] = expanding_avg.fillna(flight_mean)
        
        print(f"\nExemples de avg_delay_per_flight ({df['flight_number'].nunique()} vols uniques) :")
        sample = df.groupby("flight_number")["avg_delay_per_flight"].first().sample(10, random_state=42)
        print(sample.round(2).to_string())
        print(f"Moyenne globale : {df['avg_delay_per_flight'].mean():.2f} min")
    return df


def train_model(df, model_name: str, run_id: str):
    with mlflow.start_run(run_name=f"{model_name}_{run_id}") as run:
        print(f"\n=== Entraînement {model_name} ===")
        
        df = add_avg_delay_feature(df, is_train=True)
        
        X = df.drop(columns=["scheduled_utc", "revised_utc", "flight_number", "delay_minutes"])
        y = df["delay_minutes"]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        cat_features = [
            "aeroport_depart", "aeroport_arrivee", "terminal",
            "airline_icao", "airline_name", "aircraft_model",
            "aircraft_family", "aircraft_size_category",
            "holiday_name", "period_of_day"
        ]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostRegressor(
            iterations=12000,
            learning_rate=0.055,
            depth=8,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            early_stopping_rounds=300,
            verbose=100,
            task_type='GPU',
            l2_leaf_reg=3,
            random_strength=1.0,
            bagging_temperature=0.7
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # === ÉVALUATION ===
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)

        mlflow.log_metric("MAE", round(mae, 4))
        mlflow.log_metric("RMSE", round(rmse, 4))
        mlflow.log_metric("R2", round(r2, 4))
        mlflow.log_metric("best_iteration", model.get_best_iteration())

        # === FEATURE IMPORTANCE - UNIQUEMENT LE PLOT ===
        print("Calcul et logging du Feature Importance Plot...")
        feature_names = X_train.columns.tolist()
        importance = model.get_feature_importance()

        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values(by='importance', ascending=False)

        # Plot seulement
        plt.figure(figsize=(12, 10))
        sns.barplot(data=fi_df.head(20), x='importance', y='feature', palette="viridis")
        plt.title(f"Top 20 Feature Importance - {model_name}")
        plt.tight_layout()
        
        plt.savefig(f"{model_name}_{run_id}_feature_importance_plot.png", dpi=220, bbox_inches='tight')
        mlflow.log_artifact(f"{model_name}_{run_id}_feature_importance_plot.png", artifact_path="feature_importance")
        plt.close()

        # === ENREGISTREMENT DU MODÈLE ===
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            registered_model_name=f"CatBoost_{model_name}_Delay_Prediction"
        )

        print(f"\n=== Résultats {model_name} ===")
        print(f"MAE   : {mae:.3f} minutes")
        print(f"RMSE  : {rmse:.3f} minutes")
        print(f"R²    : {r2:.4f}")
        print(f"Best iteration : {model.get_best_iteration()}")
        print("="*70 + "\n")

        return model


def train_pipeline(run_id: str = None):
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    print(f"=== Lancement du Train Pipeline - Run ID: {run_id} ===\n")
    
    depart_bytes = load_from_s3(f"processed/train/{run_id}", f"final_departures_{run_id}.parquet")
    df_depart = pd.read_parquet(io.BytesIO(depart_bytes))
    train_model(df_depart, "Departure", run_id)
    
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