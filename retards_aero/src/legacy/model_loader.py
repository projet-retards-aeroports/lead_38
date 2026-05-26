import mlflow
import mlflow.catboost
import mlflow.sklearn
import os

model = None
te = None


_model = None  # Variable privée au module


def load_model_no_target_encoder(alias: str = "champion"):
    """Charge le modèle une seule fois et le met en cache"""
    global _model

    if _model is not None:
        return _model

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    print(f"tracking_uri = {tracking_uri}")
    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI n'est pas définie dans l'environnement"
        )

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/CatBoost_Delay_Prediction@{alias}"  # ou ppml-delay-pipeline si c'est le nouveau nom

    print(f"[INFO] Chargement du modèle depuis {model_uri} ...")
    _model = mlflow.catboost.load_model(model_uri)
    print(f"✅ Modèle chargé avec succès (@{alias})")

    return _model


def load_model_and_encoder(alias: str = "challenger"):
    global model, te

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    MODEL_NAME = "CatBoost_Delay_Prediction"

    try:
        # Chargement du modèle
        model_uri = f"models:/{MODEL_NAME}@{alias}"
        model = mlflow.catboost.load_model(model_uri)

        # Récupération du run_id pour le TargetEncoder
        client = mlflow.MlflowClient()
        version = client.get_latest_versions(MODEL_NAME, stages=["None", "Production"])[
            0
        ]
        run_id = version.run_id

        te = mlflow.sklearn.load_model(f"runs:/{run_id}/target_encoder")

        print(f"Modèle chargé avec succès (@{alias})")
        return True
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
        return False
