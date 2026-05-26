import pandas as pd
import numpy as np


def clean_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage des données pour la prédiction uniquement.
    - Garde scheduled_utc pour l'affichage dans Streamlit
    - Supprime les colonnes inutiles/leakage
    """
    df = df.copy()
    initial_shape = len(df)

    print(f" Réception de {initial_shape:,} vols pour prédiction...")

    # Colonnes à supprimer (leakage + inutiles pour le modèle)
    cols_to_drop = [
        # "flight_number",
        "airport_name",
        "revised_utc",
        "runway_utc",
        "meteo_hour_start",
        "scheduled_hour_bin",
        "meteo_hour_bin_used",
        "flight_date",
        "status",
        "mouvement_id",  # On peut le supprimer ici car on le récupère avant le nettoyage
    ]

    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Filtre critique : on doit avoir scheduled_utc
    if "scheduled_utc" in df.columns:
        before = len(df)
        df = df[df["scheduled_utc"].notna()].copy()
        removed = before - len(df)
        if removed > 0:
            print(f" → {removed:,} vols supprimés car scheduled_utc manquant")

    # ====================== NETTOYAGE COMMUN ======================
    df["terminal_dep"] = df["terminal_dep"].fillna("Unknown")
    df["terminal_arr"] = df["terminal_arr"].fillna("Unknown")

    # Correction destination_icao pour les arrivals
    if all(col in df.columns for col in ["type", "icao", "destination_icao"]):
        mask_arrival = (df["type"] == "arrival") & (df["destination_icao"].isna())
        df.loc[mask_arrival, "destination_icao"] = df.loc[mask_arrival, "icao"]

    df["destination_icao"] = df["destination_icao"].fillna("UNKNOWN")

    # Remplissage intelligent des NaN météo par aéroport
    meteo_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_gusts_10m",
        "pressure_msl",
        "precipitation",
        "cloud_cover",
    ]
    for col in meteo_cols:
        if col in df.columns:
            df[col] = df.groupby("icao")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(
                df[col].median() if not pd.isna(df[col].median()) else 0
            )

    # Features temporelles
    df["scheduled_hour"] = df["scheduled_hour"].fillna(12.0)
    df["day_of_week"] = df["day_of_week"].fillna(0.0)

    df["is_weekend"] = df["day_of_week"].isin([0, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["scheduled_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["scheduled_hour"] / 24)

    df["day_of_week"] = df["day_of_week"].astype(int)
    df["scheduled_hour"] = df["scheduled_hour"].astype(int)

    print(
        f" Nettoyage terminé | Shape : {df.shape} | NaN max = {df.isnull().mean().max():.3f}%"
    )

    return df


def clean_flight_data_pred(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage des données pour la prédiction uniquement.
    - Garde scheduled_utc pour l'affichage dans Streamlit
    - Supprime les colonnes inutiles/leakage
    """

    print(f"Shape initial du dataset : {df.shape}")
    # on ne garde que les vols futurs 'expected' et non canceled, delayed etc...
    df_predict_expected  = df[df["status"] == "Expected"].copy
    print(f"Shape du dataset après filtrage sur expected : {df_predict_expected.shape}")

    if df_predict.empty:
        print("❌ Aucun vol avec le statut 'Expected'.")
        return None, "Aucun vol prévu dans les données actuelles."

    
    # delay_minutes est supprimé du dataset de prédiction car c'est la cible à prédire, il n'est pas renseigné dans ce dataset
    df_clean = df_predict_expected.drop(
        columns=["delay_minutes"], errors="ignore"
    ).copy()

    # 2. l'api ne remplit l'aeroport de destination (destination_icao) que pour les départs car les arrivées sont deduites de la colonne icao
    # on remet l'information pour le modele
    df_clean["dest_icao_clean"] = df_clean.apply(
        lambda row: (
            row["icao"] if row["type"] == "arrival" else row["destination_icao"]
        ),
        axis=1,
    )

    # dest_icao_clean (très peu de NaN sur type departure)
    df_clean["dest_icao_clean"] = df_clean["dest_icao_clean"].fillna("UNKNOWN")

    # terminal_dep et terminal_arr (~20% NaN → on crée une vraie catégorie "inconnue")
    df_clean["terminal_dep"] = df_clean["terminal_dep"].fillna("UNKNOWN")
    df_clean["terminal_arr"] = df_clean["terminal_arr"].fillna("UNKNOWN")

    # aircraft_model (seulement 0.35% NaN → on prend la valeur la plus courante)
    most_common_model = df_clean["aircraft_model"].mode()[0]
    df_clean["aircraft_model"] = df_clean["aircraft_model"].fillna(most_common_model)

    # Traitement des valeurs manquantes restantes

    # a) Variables météo => imputation par la médiane
    meteo_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_gusts_10m",
        "pressure_msl",
        "precipitation",
        "cloud_cover",
    ]

    for col in meteo_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # on ne gère plus delay_minutes (cible absente du dataset de prédiction)

    df_fe = df_clean.copy()

    # 1. Convertir en datetime
    df_fe["scheduled_utc"] = pd.to_datetime(df_fe["scheduled_utc"])

    # 2. Features temporelles de base SUR LE MOUVEMENT
    df_fe["dep_hour"] = df_fe["scheduled_utc"].dt.hour
    df_fe["dep_dayofweek"] = df_fe["scheduled_utc"].dt.dayofweek
    df_fe["dep_dayofmonth"] = df_fe["scheduled_utc"].dt.day
    df_fe["dep_month"] = df_fe["scheduled_utc"].dt.month

    # 3. Encodage cyclique
    df_fe["dep_hour_sin"] = np.sin(2 * np.pi * df_fe["dep_hour"] / 24)
    df_fe["dep_hour_cos"] = np.cos(2 * np.pi * df_fe["dep_hour"] / 24)

    df_fe["dep_dayofweek_sin"] = np.sin(2 * np.pi * df_fe["dep_dayofweek"] / 7)
    df_fe["dep_dayofweek_cos"] = np.cos(2 * np.pi * df_fe["dep_dayofweek"] / 7)

    # 4. Période de la journée
    df_fe["period_of_day"] = pd.cut(
        df_fe["dep_hour"],
        bins=[0, 6, 10, 16, 20, 24],
        labels=[
            "night",
            "morning",
            "afternoon",
            "evening",
            "late_night",
        ],  # 'night' → 'late_night'
        right=False,
        ordered=False,  # Important labels identiques
    )

    print("Feature Engineering terminé.")
    print(f"Shape final : {df_fe.shape}")
    print("Colonnes ajoutées : features temporelles")

    # Suppression des colonnes inutiles ou trop incomplètes (optionnel, à ajuster selon les besoins)
    colonnes_a_dropper = [
        "scheduled_utc",
        "flight_number",
        "date",
        "destination_icao",
        "destination_iata",
        "destination_name",
        "holiday_name",
        "runway_utc",
        "call_sign",
        "aircraft_reg",
        "aircraft_mode_s",
        "airline_iata",
        "airline_icao",
        "is_weekend_or_holiday",
        # leakage potentiel
        "status",
        "delay_minutes",  # deja dropée mais on applique le meme drop que sur le train par principe de précaution
        "revised_utc",
        "runway_utc",
        "gate_dep",
        "baggage_belt",
        "quality_dep",
        "quality_arr",
    ]

    df_fe = df_fe.drop(columns=colonnes_a_dropper, errors="ignore")
    print(
        f"✅ Nettoyage terminé | Shape : {df.shape} | NaN max = {df.isnull().mean().max():.3f}%"
    )

    return df_fe


def prepare_for_model(df_clean: pd.DataFrame, te) -> pd.DataFrame:
    """
    Applique le Target Encoding + nettoyage final des features catégorielles.
    Cette étape doit être faite JUSTE avant la prédiction.
    """
    df = df_clean.copy()

    high_card_cols = ["airline_name", "dest_icao_clean", "aircraft_model"]

    # Target Encoding
    X_pred = te.transform(df)
    X_pred = X_pred.drop(columns=high_card_cols, errors="ignore")

    # Nettoyage final des variables catégorielles pour CatBoost
    categorical_features = [
        "icao",
        "type",
        "codeshare_status",
        "terminal_dep",
        "terminal_arr",
        "aircraft_family",
        "aircraft_size_category",
        "period_of_day",
    ]

    for col in categorical_features:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].fillna("MISSING").astype(str)

    return X_pred
