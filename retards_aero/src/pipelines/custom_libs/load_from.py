import boto3
import os

s3 = boto3.client('s3')


def load_from_s3(folder: str, filename: str) -> bytes | None:

    """Téléchargement simple depuis S3
    
    en dehors de la fonction 
    # Chargement
    data_bytes = load_from_s3("raw/aeroports/historical", "vols_2026-05-24.json")

    # Si c'est du JSON
    if data_bytes:
        import json
        data = json.loads(data_bytes)

    # Si c'est du Parquet
    if data_bytes:
        import pandas as pd
        import io
        df = pd.read_parquet(io.BytesIO(data_bytes))

    """
    key = f"projet_final_lead/{folder}/{filename}"
    try:
        response = s3.get_object(Bucket=os.getenv("BUCKET"), Key=key)
        print(f"S3 <- {key}")
        return response['Body'].read()
    except Exception as e:
        print(f"Erreur dans load_from_s3 {key} : {e}")
        return None
