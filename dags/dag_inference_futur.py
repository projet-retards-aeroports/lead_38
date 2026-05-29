from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# Configuration par défaut (incluant l'alerte mail demandée)
default_args = {
    'owner': 'MLOps_Surel',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 1),
    'email': ['projet_final_lead38@proton.me'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

PROJET_PATH = r"C:\Users\surel\Desktop\Formation_DATA_JEDHA\projet-retard-aero"

def sauvegarder_run_id_context(**kwargs):
    """Sauvegarde le run_id Airflow actuel dans une variable globale Airflow"""
    current_run_id = kwargs['run_id']
    Variable.set("dernier_run_prediction", current_run_id)
    print(f"✓ Variable Airflow 'dernier_run_prediction' mise à jour avec : {current_run_id}")

with DAG(
    'dag_inference_futur_hebdo',
    default_args=default_args,
    description='Pipeline hebdomadaire de récupération J+7 et prédiction des retards',
    schedule='30 3 * * 0',  # Tous les dimanches à 3h30
    catchup=False,
    tags=['production', 'inference'],
) as dag:

    # 1. Requêtage des API (Vols et Météo en parallèle)
    fetch_vols = BashOperator(
        task_id='fetch_vols_futurs',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/fetch_vols_raw.py --run_id '
                     f'"{{{{ run_id }}}}" --is_future',
    )

    fetch_meteo = BashOperator(
        task_id='fetch_meteo_futurs',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/fetch_meteo_raw.py --run_id '
                     f'"{{{{ run_id }}}}" --is_future',
    )

    # 2. Fusion et préparation des datasets de base
    merge_data = BashOperator(
        task_id='transform_merge_vols_meteo',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_merge_vols_meteo.py --run_id '
                     f'"{{{{ run_id }}}}" --is_future',
    )

    extract_departures = BashOperator(
        task_id='extract_departure_dataset',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_departure_dataset.py --run_id '
                     f'"{{{{ run_id }}}}" --is_future',
    )

    extract_arrivals = BashOperator(
        task_id='extract_arrivals_dataset',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_arrivals_dataset.py --run_id '
                     f'"{{{{ run_id }}}}" --is_future',
    )

    # 3. Nettoyage, Imputation et Features Engineering
    clean_datasets = BashOperator(
        task_id='transform_clean',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_clean.py --run_id '
                     f'"{{{{ run_id }}}}" --is_future',
    )

    impute_datasets = BashOperator(
        task_id='transform_imputation',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_imputation.py --run_id '
                     f'"{{{{ run_id }}}}" --is_future',
    )

    feature_engineering = BashOperator(
        task_id='transform_features',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_features.py --run_id '
                     f'"{{{{ run_id }}}}" --is_future',
    )

    # 4. Inférence (Calcul des prédictions à partir des modèles champions)
    run_inference = BashOperator(
        task_id='run_inference_pipeline',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/inference_pipeline.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    # 5. Enregistrement du run_id pour le suivi quotidien
    update_prediction_variable = PythonOperator(
        task_id='update_last_prediction_run_id',
        python_callable=sauvegarder_run_id_context,
    )

    # Définition des dépendances du pipeline
    [fetch_vols, fetch_meteo] >> merge_data
    merge_data >> [extract_departures, extract_arrivals] >> clean_datasets
    clean_datasets >> impute_datasets >> feature_engineering >> run_inference >> update_prediction_variable