import pandas as pd
import numpy as np
from catboost import Pool

# from src.model_loader import model, te
from src.preprocessing import clean_flight_data_pred, prepare_for_model


def predict_delays(df_raw: pd.DataFrame, model) -> pd.DataFrame:
    """Fonction principale de prédiction avec CatBoost (sans TargetEncoder)"""

    if model is None:
        raise Exception("Modèle CatBoost non fourni à predict_delays()")

    try:
        print(f"[predict_delays] {df_raw.shape[0]} lignes reçues initialement")

        # === 1. Nettoyage + Feature Engineering ===
        df_clean = clean_flight_data_pred(df_raw)
        print(
            f"[predict_delays] Après nettoyage → {df_clean.shape[0]} lignes conservées"
        )

        # === 2. Sauvegarde des colonnes d'affichage (avant modification) ===
        display_cols = ["flight_number", "scheduled_utc"]
        display_data = None
        if all(col in df_raw.columns for col in display_cols):
            surviving_indices = df_clean.index
            display_data = df_raw.loc[surviving_indices, display_cols].copy()
            print(
                f"[predict_delays] Colonnes d'affichage sauvegardées pour {len(display_data)} lignes"
            )

        # === 3. Préparation pour le modèle (PLUS DE TE !) ===
        # On passe directement les données nettoyées
        X_pred = (
            df_clean.copy()
        )  # ou prepare_for_model(df_clean) si tu as d'autres transformations

        # === 4. Features catégorielles (ajoute les 3 colonnes qui étaient encodées avant) ===
        categorical_features = [
            "icao",
            "type",
            "codeshare_status",
            "terminal_dep",
            "terminal_arr",
            "aircraft_family",
            "aircraft_size_category",
            "period_of_day",
            "airline_name",  # plus de TE
            "dest_icao_clean",  # idem
            "aircraft_model",  # idem
        ]

        # Filtrer uniquement les colonnes qui existent vraiment
        cat_features_present = [c for c in categorical_features if c in X_pred.columns]

        # === 5. Création du Pool et Prédiction ===
        pred_pool = Pool(X_pred, cat_features=cat_features_present)

        predictions = model.predict(pred_pool)

        # === 6. Construction du résultat final ===
        df_result = df_clean.copy()
        df_result["predicted_delay_minutes"] = np.round(predictions, 2)

        # Ajout des colonnes d'affichage
        if display_data is not None:
            df_result["flight_number"] = display_data["flight_number"].values
            df_result["scheduled_utc"] = display_data["scheduled_utc"].values

        print(f"[predict_delays] Prédiction finale réussie : {len(predictions)} vols")
        print(f"Colonnes envoyées vers Streamlit : {df_result.columns.tolist()}")

        return df_result

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise Exception(f"Erreur pendant la prédiction : {str(e)}") from e


def predict_delays_old(df_raw: pd.DataFrame, model=None, te=None) -> pd.DataFrame:
    """Fonction principale de prédiction"""
    if model is None or te is None:
        raise Exception("Modèle ou TargetEncoder non fourni à predict_delays()")

    try:
        print(f"[predict_delays] {df_raw.shape[0]} lignes reçues initialement")

        # === 1. Nettoyage + Feature Engineering EN PREMIER ===
        df_clean = clean_flight_data_pred(df_raw)
        print(
            f"[predict_delays] Après nettoyage → {df_clean.shape[0]} lignes conservées"
        )

        # === 2. Sauvegarder les colonnes d'affichage APRÈS nettoyage ===
        # On utilise l'index restant après nettoyage
        display_cols = ["flight_number", "scheduled_utc"]
        display_data = None

        if all(col in df_raw.columns for col in display_cols):
            # On récupère uniquement les lignes qui ont survécu au nettoyage
            surviving_indices = df_clean.index
            display_data = df_raw.loc[surviving_indices, display_cols].copy()
            print(
                f"[predict_delays] Colonnes d'affichage sauvegardées pour {len(display_data)} lignes"
            )

        # 3. Préparation pour le modèle
        X_pred = prepare_for_model(df_clean, te)

        # 4. Prédiction
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

        pred_pool = Pool(
            X_pred,
            cat_features=[c for c in categorical_features if c in X_pred.columns],
        )
        predictions = model.predict(pred_pool)

        # 5. Construction du résultat
        df_result = df_clean.copy()
        df_result["predicted_delay_minutes"] = np.round(predictions, 2)

        # 6. Ajout des colonnes d'affichage (version sûre)
        if display_data is not None:
            df_result["flight_number"] = display_data["flight_number"].values
            df_result["scheduled_utc"] = display_data["scheduled_utc"].values
            print(
                f"[predict_delays] Colonnes flight_number et scheduled_utc ajoutées avec succès"
            )
        print(f"colonnes envoyées vers streamlit: {df_result.columns}")
        print(f"[predict_delays] Prédiction finale réussie : {len(predictions)} vols")
        return df_result

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise Exception(f"Erreur pendant la prédiction : {str(e)}") from e


def predict_delays_old(df_raw: pd.DataFrame, model=None, te=None) -> pd.DataFrame:
    """Fonction principale de prédiction"""

    if model is None or te is None:
        raise Exception("Modèle ou TargetEncoder non fourni à predict_delays()")

    try:
        print(f"[predict_delays] {df_raw.shape[0]} lignes reçues")

        # === 1. Sauvegarder les colonnes d'affichage AVEC leur index d'origine ===
        display_cols = ["flight_number", "scheduled_utc"]
        display_data = None

        if all(col in df_raw.columns for col in display_cols):
            display_data = df_raw[display_cols].copy()

        # 1. Nettoyage + Feature Engineering
        df_clean = clean_flight_data_pred(df_raw)

        # 2. Préparation pour le modèle
        X_pred = prepare_for_model(df_clean, te)

        # 3. Prédiction
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

        pred_pool = Pool(
            X_pred,
            cat_features=[c for c in categorical_features if c in X_pred.columns],
        )

        predictions = model.predict(pred_pool)

        df_result = df_clean.copy()
        df_result["predicted_delay_minutes"] = np.round(predictions, 2)

        # 6. Réajouter les colonnes d'affichage en respectant l'ordre des lignes
        if display_data is not None:
            # On réindexe sur l'index original pour garantir le bon alignement
            df_result = df_result.join(display_data, how="left")
        print(f"DEBUG : {df_result.columns}")
        print(f"[predict_delays] Prédiction réussie : {len(predictions)} vols")
        return df_result

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise Exception(f"Erreur pendant la prédiction : {str(e)}") from e
