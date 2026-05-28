import argparse
from datetime import datetime
import subprocess
import uuid
from dotenv import load_dotenv
load_dotenv()

def run_full_pipeline(run_id: str = None):
    if run_id is None:
        run_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    print(f"=== Pipeline complet - Run ID: {run_id} ===\n")

    scripts = [
        #("fetch_vols_raw.py", "Récupération des vols"),
        #("fetch_meteo_raw.py", "Récupération de la météo"),
        #("transform_merge_vols_meteo.py", "Fusion vols + météo"),
        ("transform_departure_dataset.py",  "Dataset départs"),
        ("transform_arrivals_dataset.py",   "Dataset arrivées"),
    ]

    for script, desc in scripts:
        print(f"→ {desc}...")
        subprocess.run(["python", script, "--run_id", run_id], check=True)
        print(f"   ✓ Terminé\n")

    print(f"=== Pipeline terminé ===")
    print(f"Run ID : {run_id}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline complet")
    parser.add_argument("--run_id", type=str, default=None, help="Optionnel : forcer un run_id")
    args = parser.parse_args()

    run_full_pipeline(args.run_id)


if __name__ == "__main__":
    main()