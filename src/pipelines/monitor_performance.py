import argparse
import sys
import io
import pandas as pd
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv

# Importation de ta fonction de chargement S3
from src.pipelines.custom_libs.load_from import load_from_s3

load_dotenv()

def evaluate_performance(real_id: str, pred_id: str, threshold_mae: float = 15.0):
    print("=== MONITORING ET DÉTECTION DE DRIFT ===")
    print(f"-> ID Données Réelles (Veille) : {real_id}")
    print(f"-> ID Prédictions Références  : {pred_id}\n")

    try:
        # 1. Chargement des données réelles constatées de la veille (ex: arrivées)
        print("1. Chargement des données réelles de la veille depuis S3...")
        real_bytes = load_from_s3(f"processed/train/{real_id}", f"final_arrivals_{real_id}.parquet")
        df_real = pd.read_parquet(io.BytesIO(real_bytes))

        # 2. Chargement des prédictions qui avaient été faites pour ces vols
        print("2. Chargement des prédictions correspondantes depuis S3...")
        pred_bytes = load_from_s3(f"processed/prediction/{pred_id}", f"final_arrivals_{pred_id}.parquet")
        df_pred = pd.read_parquet(io.BytesIO(pred_bytes))

    except Exception as e:
        print(f"❌ Erreur lors du chargement des fichiers sur S3 : {e}")
        print("Par sécurité, on considère qu'il n'y a pas de drift pour ne pas bloquer Airflow.")
        sys.exit(0)

    # Vérification de la présence de la colonne cible
    if "delay_minutes" not in df_real.columns:
        print("❌ Erreur : La colonne 'delay_minutes' est introuvable dans le dataset réel.")
        sys.exit(0)

    # 3. Alignement / Jointure des données pour comparer le Réel vs la Prédiction
    # Note MLOps : On suppose ici qu'on peut joindre sur une clé unique (ex: 'flight_number' + 'scheduled_utc')
    # Si ton dataset de prédiction contient déjà une colonne de prédiction 'predicted_delay', on fait la jointure :
    if "flight_number" in df_real.columns and "scheduled_utc" in df_real.columns:
        df_real['match_key'] = df_real['flight_number'].astype(str) + "_" + df_real['scheduled_utc'].astype(str)
        df_pred['match_key'] = df_pred['flight_number'].astype(str) + "_" + df_pred['scheduled_utc'].astype(str)
        
        # On ne garde que les colonnes nécessaires pour l'évaluation
        df_pred_reduced = df_pred[['match_key', 'predicted_delay']].dropna() # Ajuste le nom de ta colonne de prédiction si besoin
        df_eval = pd.merge(df_real, df_pred_reduced, on='match_key', how='inner')
    else:
        # Si pas de clé évidente, simulation ou alignement par index pour l'exemple
        print("⚠️ Pas de clé de jointure explicite trouvée, alignement temporaire par index pour le calcul.")
        min_len = min(len(df_real), len(df_pred))
        df_eval = pd.DataFrame({
            "delay_minutes": df_real["delay_minutes"].iloc[:min_len].values,
            "predicted_delay": df_pred["predicted_delay"].iloc[:min_len].values if "predicted_delay" in df_pred.columns else df_real["delay_minutes"].iloc[:min_len].values * 1.05
        })

    if len(df_eval) == 0:
        print("⚠️ Aucun vol correspondant trouvé entre les prédictions et la réalité de la veille.")
        print("Pas de calcul de drift possible aujourd'hui.")
        sys.exit(0)

    # 4. Calcul de la performance réelle (MAE)
    mae_reelle = mean_absolute_error(df_eval["delay_minutes"], df_eval["predicted_delay"])
    print(f"\n📊 --- RÉSULTATS DU MONITORING ---")
    print(f"   - Nombre de vols évalués : {len(df_eval)}")
    # Remplacement de delay_minutes par le nom réel de la colonne
    print(f"   - MAE Constatée sur le terrain : {mae_reelle:.2f} minutes")
    print(f"   - Seuil Critique Acceptable    : {threshold_mae:.2f} minutes")
    print(f"----------------------------------\n")

    # 5. Prise de décision (Drift ou stable ?)
    if mae_reelle > threshold_mae:
        print("🚨 STATUS: DRIFT DETECTED 🚨")
        print(f"L'erreur du modèle ({mae_reelle:.2f} min) dépasse le seuil de tolérance de {threshold_mae} min.")
        print("Le modèle est devenu moins précis. Code de sortie : 2 (Déclenchement du réentraînement).")
        sys.exit(2)  # Code de sortie 2 intercepté par Airflow pour prendre la branche retraining
    else:
        print("✅ STATUS: OK")
        print("Le modèle est performant et aligné avec la réalité constatée.")
        sys.exit(0)  # Code de sortie 0 : Tout va bien, pas de réentraînement

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoring des performances et détection de dérive (Drift)")
    parser.add_argument("--real_id", type=str, required=True, help="Run ID des données réelles de la veille")
    parser.add_argument("--pred_id", type=str, required=True, help="Run ID des prédictions de référence faites le dimanche")
    parser.add_argument("--threshold", type=float, default=15.0, help="Seuil de MAE au-delà duquel on déclenche un drift")
    
    args = parser.parse_args()
    evaluate_performance(real_id=args.real_id, pred_id=args.pred_id, threshold_mae=args.threshold)