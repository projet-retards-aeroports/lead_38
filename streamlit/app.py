import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import boto3
from botocore.exceptions import ClientError
import io
import requests

# ==================== CONFIG ====================
load_dotenv()
st.set_page_config(page_title="Prédiction Retards Aériens", layout="wide")
st.title("✈️ Prédiction des Retards Aériens")

# ==================== SIDEBAR ====================
st.sidebar.header("Configuration")

run_id_future = "2026-05-25_221052_8f2d98"
st.sidebar.write("**Run ID Future :**", run_id_future)

flight_type = st.sidebar.selectbox("Type de vol", ["departures", "arrivals"])

# --- DATE + HEURE ---
st.sidebar.subheader("📅 Filtre Temporel")

selected_date = st.sidebar.date_input("Date du vol")
selected_time = st.sidebar.time_input("Heure (optionnel)", value=None)

# --- COMPAGNIES (autocomplétion) ---
st.sidebar.subheader("✈️ Compagnies")

# ==================== CHARGEMENT DES DONNÉES ====================
@st.cache_data(ttl=300)
def load_from_s3(key: str) -> pd.DataFrame:
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
        )
        bucket = os.getenv("BUCKET")
        response = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(response["Body"].read()))
    except ClientError as e:
        st.error(f"Erreur S3 : {e}")
        return pd.DataFrame()

key = f"projet_final_lead/processed/prediction/{run_id_future}/clean_{flight_type}_{run_id_future}.parquet"
df = load_from_s3(key)

if df.empty:
    st.stop()

# ==================== FILTRES ====================
st.subheader("Filtres")

df["scheduled_utc"] = pd.to_datetime(df["scheduled_utc"], errors="coerce")

min_date = df["scheduled_utc"].min().date()
max_date = df["scheduled_utc"].max().date()

# Appliquer les filtres
df_filtered = df[df["scheduled_utc"].dt.date == selected_date].copy()

if selected_time:
    df_filtered = df_filtered[df_filtered["scheduled_utc"].dt.time == selected_time].copy()

# Filtre compagnies
airlines = st.sidebar.multiselect(
    "Choisir les compagnies",
    options=sorted(df["airline_name"].dropna().unique())
)

if airlines:
    df_filtered = df_filtered[df_filtered["airline_name"].isin(airlines)].copy()

st.write(f"**{len(df_filtered)} vols** disponibles")

# ==================== TABLEAU AVEC CHECKBOX EN PREMIÈRE COLONNE ====================
if df_filtered.empty:
    st.warning("Aucun vol trouvé pour ces filtres.")
    st.stop()

cols_to_keep = ["flight_number", "scheduled_utc", "aeroport_depart", "aeroport_arrivee"]

df_select = df_filtered[cols_to_keep].copy()
df_select.insert(0, "Select", False)   # Case à cocher en 1ère colonne

edited_df = st.data_editor(
    df_select,
    column_config={
        "Select": st.column_config.CheckboxColumn("✓", default=False, width="small")
    },
    hide_index=True,
    use_container_width=True,
    key="flight_table"
)

selected_df = edited_df[edited_df["Select"] == True]

if not selected_df.empty:
    df_to_pred = df_filtered.loc[selected_df.index].copy()
else:
    df_to_pred = pd.DataFrame(columns=cols_to_keep)

st.write(f"**{len(df_to_pred)} vols sélectionnés**")

if not df_to_pred.empty:
    st.dataframe(df_to_pred[cols_to_keep].head(10), use_container_width=True)

# ==================== PRÉDICTION ====================
if st.button("🚀 Lancer les Prédictions", type="primary"):
    if df_to_pred.empty:
        st.warning("Veuillez sélectionner au moins un vol.")
        st.stop()

    with st.spinner("Prédiction en cours via l'API..."):
        payload = {
            "data": df_to_pred.to_dict(orient="records"),
            "flight_type": flight_type,
            "run_id": run_id_future
        }

        try:
            api_base_url = os.getenv("api_base_url")
            response = requests.post(f"{api_base_url}/predict", json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                predictions = result["predictions"]

                df_result = df_to_pred.copy()
                df_result["predicted_delay_minutes"] = predictions

                st.success(f"✅ Prédictions terminées pour {len(predictions)} vols !")

                cols_to_show = cols_to_keep + ["predicted_delay_minutes"]
                st.dataframe(
                    df_result[cols_to_show].sort_values("predicted_delay_minutes", ascending=False),
                    use_container_width=True
                )

                # Téléchargement
                csv = df_result[cols_to_show].to_csv(index=False)
                st.download_button(
                    "📥 Télécharger les prédictions (CSV)",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
            else:
                st.error(f"Erreur API : {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")
