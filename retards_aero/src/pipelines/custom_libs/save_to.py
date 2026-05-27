import boto3
import os

from dotenv import load_dotenv
load_dotenv()
s3 = boto3.client('s3')


def save_to_s3(body: bytes, folder: str, filename: str) -> bool:
    """Upload simple vers S3
    
    # Exemple JSON
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    save_to_s3(json_bytes, "raw/aeroports/historical", "vols_2026-05-24.json")

    # Exemple Parquet
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, compression="gzip")
    save_to_s3(buffer.getvalue(), "processed/train", "dataset_train_2026-05-24.parquet")


    """
    key = f"projet_final_lead/{folder}/{filename}"
    try:
        s3.put_object(
            Bucket=os.getenv("BUCKET"),
            Key=key,
            Body=body
        )
        print(f"S3 -> {key}")
        return True
    except Exception as e:
        print(f"Erreur S3 {key} : {e}")
        return False
