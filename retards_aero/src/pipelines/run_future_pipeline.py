import argparse
from datetime import datetime
import subprocess
import uuid


def run_future_pipeline(run_id: str = None):
    if run_id is None:
        run_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    print(f"=== Pipeline FUTUR - Run ID: {run_id} ===\n")
    print("Objectif : Récupérer données J+1 à J+8 pour prédictions utilisateur\n")

    scripts = [
        #("fetch_vols_raw.py",          "Récupération vols futurs (J+1 à J+8)"),
        #("fetch_meteo_raw.py",         "Récupération météo future (J+0 à J+8)"),
        #("transform_merge_vols_meteo.py", "Fusion vols + météo futurs"),
        ("transform_departure_dataset.py", "Dataset final Départs futur"),
        ("transform_arrivals_dataset.py",  "Dataset final Arrivées futur"),
    ]

    for script, desc in scripts:
        print(f"→ {desc}...")
        # On passe un flag pour indiquer que c'est du futur
        subprocess.run([
            "python", 
            script, 
            "--run_id", run_id,
            "--is_future"   # ← Flag important
        ], check=True)
        print(f"   ✓ Terminé\n")

    print(f"=== Pipeline FUTUR terminé ===")
    print(f"Run ID : {run_id}\n")
    print(f"Données futures prêtes pour l'inférence utilisateur.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline données FUTURES")
    parser.add_argument("--run_id", type=str, default=None, help="Optionnel : forcer un run_id")
    args = parser.parse_args()

    run_future_pipeline(args.run_id)


if __name__ == "__main__":
    main()