import argparse
from datetime import datetime
import subprocess
import uuid
from dotenv import load_dotenv
load_dotenv()

def run_pipeline(run_id: str = None, is_future: bool = False):
    if run_id is None:
        run_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    mode = "FUTUR" if is_future else "TRAIN"
    print(f"=== Pipeline {mode} - Run ID: {run_id} ===\n")
    if is_future:
        print("Objectif : Récupérer données J+1 à J+8 pour prédictions\n")

    scripts = [
        
        #("fetch_vols_raw.py",          "Récupération vols" + (" futurs (J+1 à J+8)" if is_future else "")),
        #("fetch_meteo_raw.py",         "Récupération météo" + (" future (J+0 à J+8)" if is_future else "")),
        ("transform_merge_vols_meteo.py", "Fusion vols + météo" + (" futurs" if is_future else "")),
        ("transform_departure_dataset.py", "Dataset départs" + (" futurs" if is_future else "")),
        ("transform_arrivals_dataset.py", "Dataset arrivées" + (" futurs" if is_future else "")),
    ]

    for script, desc in scripts:
        print(f"→ {desc}...")
        cmd = ["python", script, "--run_id", run_id]
        if is_future:
            cmd.append("--is_future")
        subprocess.run(cmd, check=True)
        print("   ✓ Terminé\n")

    print(f"=== Pipeline {mode} terminé ===")
    print(f"Run ID : {run_id}")
    if is_future:
        print("Données futures prêtes pour l'inférence utilisateur.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline complet ou futur")
    parser.add_argument("--run_id", type=str, default=None, help="Optionnel : forcer un run_id")
    parser.add_argument("--future", action="store_true", help="Mode futur (J+1 à J+8)")
    args = parser.parse_args()
    run_pipeline(args.run_id, args.future)


if __name__ == "__main__":
    main()