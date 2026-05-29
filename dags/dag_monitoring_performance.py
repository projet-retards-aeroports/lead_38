from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable

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

def evaluer_drift_et_choisir_branche(**kwargs):
    """
    Récupère le run_id du dimanche pour trouver les prédictions associées,
    exécute le script de monitoring et analyse son résultat pour décider s'il y a drift.
    """
    # Récupération sécurisée du run_id du dimanche dernier
    try:
        run_id_pred = Variable.get("dernier_run_prediction")
    except KeyError:
        print("Attention : Aucune variable 'dernier_run_prediction' trouvée. Annulation.")
        return 'pas_de_drift_constate'
        
    current_run_id = kwargs['run_id']
    
    print(f"Comparaison des données réelles du run {current_run_id} avec les prédictions du run {run_id_pred}")
    
    # Ici, on simule l'appel ou le retour de ton script de monitoring.
    # Ton script de monitoring peut écrire un statut dans un fichier texte ou lever une exception spécifique.
    # Disons que ton script monitor_performance.py renvoie un code de sortie 0 si OK, et 2 si DRIFT.
    
    import subprocess
    cmd = [
        "python", 
        f"{PROJET_PATH}/src/pipelines/monitor_performance.py", 
        "--real_id", current_run_id, 
        "--pred_id", run_id_pred
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    # On imagine que ton script affiche 'STATUS: DRIFT' en cas de dérive
    if "STATUS: DRIFT" in result.stdout or result.returncode == 2:
        print("🚨 Dérive du modèle constatée ! Redirection vers la branche de réentraînement.")
        return 'declencher_retrain_pipeline'
    else:
        print("✅ Performances stables. Pas de réentraînement nécessaire.")
        return 'pas_de_drift_constate'

with DAG(
    'dag_monitoring_performance_quotidien',
    default_args=default_args,
    description='Calcul des performances réelles quotidiennes et gestion du drift',
    schedule='0 1 * * *',  # Tous les jours à 1h00
    catchup=False,
    tags=['production', 'monitoring'],
) as dag:

    # 1. Récupération des données réelles de la veille (Historique = is_future False)
    fetch_vols_reels = BashOperator(
        task_id='fetch_vols_reels_veille',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/fetch_vols_raw.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    fetch_meteo_reelle = BashOperator(
        task_id='fetch_meteo_reelle_veille',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/fetch_meteo_raw.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    # 2. Pipeline de transformation des données réelles constatées
    merge_reels = BashOperator(
        task_id='transform_merge_reels',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_merge_vols_meteo.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    extract_departures_reels = BashOperator(
        task_id='extract_departure_reels',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_departure_dataset.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    extract_arrivals_reels = BashOperator(
        task_id='extract_arrivals_reels',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_arrivals_dataset.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    clean_reels = BashOperator(
        task_id='transform_clean_reels',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_clean.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    impute_reels = BashOperator(
        task_id='transform_imputation_reels',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_imputation.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    feature_engineering_reels = BashOperator(
        task_id='transform_features_reels',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/transform_features.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    # 3. Évaluation de la performance réelle vs Prédictions (Routeur/Branchement)
    check_drift = BranchPythonOperator(
        task_id='check_performance_and_drift',
        python_callable=evaluer_drift_et_choisir_branche,
    )

    # Branche A : Drift constaté -> Réentraînement & Log MLflow
    # (Le script train_pipeline.py devra inclure la logique pour tagguer en Challenger sur MLflow)
    retrain_model = BashOperator(
        task_id='declencher_retrain_pipeline',
        bash_command=f'cd {PROJET_PATH} && python src/pipelines/train_pipeline.py --run_id '
                     f'"{{{{ run_id }}}}"',
    )

    # Branche B : Pas de drift -> On s'arrête proprement
    no_drift = EmptyOperator(
        task_id='pas_de_drift_constate',
    )

    # Logique d'enchaînement
    [fetch_vols_reels, fetch_meteo_reelle] >> merge_reels
    merge_reels >> [extract_departures_reels, extract_arrivals_reels] >> clean_reels
    clean_reels >> impute_reels >> feature_engineering_reels >> check_drift
    
    # Branchements issus du BranchPythonOperator
    check_drift >> retrain_model
    check_drift >> no_drift