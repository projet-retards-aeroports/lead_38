import argparse
import pandas as pd
import io
import numpy as np
from datetime import datetime
import holidays
from vacances_scolaires_france import SchoolHolidayDates

from src.load_from import load_from_s3
from src.save_to import save_to_s3


def enrich_aircraft_features(df: pd.DataFrame) -> pd.DataFrame:
    """Version optimisée avec np.select()"""
    df = df.copy()
    f = df["aircraft_model"].fillna("Other").astype(str).str.upper().str.strip()

    conditions = [
        f.str.contains("A320|A319|A321|A318") & (f.str.contains("NEO|SHARKLETS")),
        f.str.contains("A320|A319|A321|A318"),
        f.str.contains("A350"),
        f.str.contains("A330"),
        f.str.contains("A380"),
        f.str.contains("A220|CS300"),
        f.str.contains("B737|737") & f.str.contains("MAX"),
        f.str.contains("B737|737"),
        f.str.contains("B777|777"),
        f.str.contains("B787|787"),
        f.str.contains("E190|E195|E170|E175|ERJ|EMB"),
        f.str.contains("ATR 72"),
        f.str.contains("ATR 42"),
        f.str.contains("CRJ"),
        f.str.contains("Q400|DHC-8"),
        f.str.contains("B767"),
        f.str.contains("B757"),
        f.str.contains("B747"),
    ]

    choices = [
        "Airbus A320 NEO", "Airbus A320 Family", "Airbus A350", "Airbus A330",
        "Airbus A380", "Airbus A220", "Boeing 737 MAX", "Boeing 737 Family",
        "Boeing 777", "Boeing 787", "Embraer E-Jet", "ATR 72", "ATR 42",
        "Bombardier CRJ", "Bombardier Dash 8 Q400", "Boeing 767", "Boeing 757", "Boeing 747"
    ]

    df["aircraft_family"] = np.select(conditions, choices, default="Other")

    seats_mapping = {
        "Airbus A320 Family": 165, "Airbus A320 NEO": 180, "Airbus A220": 135,
        "Boeing 737 Family": 155, "Boeing 737 MAX": 178, "Boeing 777": 355,
        "Boeing 787": 295, "Airbus A350": 325, "Airbus A330": 275,
        "Airbus A380": 525, "Embraer E-Jet": 100, "ATR 72": 68,
        "ATR 42": 48, "Bombardier CRJ": 85, "Bombardier Dash 8 Q400": 78,
        "Boeing 767": 260, "Boeing 757": 200, "Boeing 747": 410, "Other": 150,
    }

    df["num_seats"] = df["aircraft_family"].map(seats_mapping).fillna(150).astype(int)
    df["is_widebody"] = df["aircraft_family"].isin(["Airbus A330","Airbus A350","Airbus A380","Boeing 777","Boeing 787","Boeing 767","Boeing 747"])
    df["is_narrowbody"] = df["aircraft_family"].isin(["Airbus A320 Family","Airbus A220","Boeing 737 Family","Boeing 737 MAX","Embraer E-Jet"])
    df["is_regional"] = df["aircraft_family"].isin(["ATR 72","ATR 42","Bombardier CRJ","Bombardier Dash 8 Q400","Embraer E-Jet"])

    def get_size_category(family):
        if family in ["Airbus A380","Boeing 777","Boeing 787","Airbus A350","Boeing 747"]:
            return "Very Large"
        elif family in ["Airbus A330","Boeing 767"]:
            return "Large"
        elif family in ["Airbus A320 Family","Boeing 737 Family","Boeing 737 MAX","Airbus A220"]:
            return "Medium"
        else:
            return "Small"

    df["aircraft_size_category"] = df["aircraft_family"].apply(get_size_category)
    df["is_freighter"] = False

    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features jours fériés et vacances scolaires"""
    df = df.copy()
    df["scheduled_utc"] = pd.to_datetime(df["scheduled_utc"], errors="coerce")
    df["flight_date"] = df["scheduled_utc"].dt.date

    fr_holidays = holidays.FR(years=range(2024, 2029))
    dte_schools = SchoolHolidayDates()

    df["is_holiday"] = df["flight_date"].apply(lambda x: 1 if x in fr_holidays else 0)
    df["vac_school"] = df["flight_date"].apply(lambda x: 1 if dte_schools.is_holiday(x) else 0)
    df["is_holiday_eve"] = df["flight_date"].apply(lambda x: 1 if (x + pd.Timedelta(days=1)) in fr_holidays else 0)
    df["is_holiday_next"] = df["flight_date"].apply(lambda x: 1 if (x - pd.Timedelta(days=1)) in fr_holidays else 0)
    df["is_weekend"] = df["scheduled_utc"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_weekend_or_holiday"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    df["holiday_name"] = df["flight_date"].apply(lambda x: fr_holidays.get(x, "no_holiday"))

    df = df.drop(columns=["flight_date"], errors="ignore")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features temporelles"""
    df = df.copy()
    df["scheduled_utc"] = pd.to_datetime(df["scheduled_utc"], errors="coerce")

    df["hour"] = df["scheduled_utc"].dt.hour
    df["day_of_week"] = df["scheduled_utc"].dt.dayofweek
    df["month"] = df["scheduled_utc"].dt.month
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["is_morning_peak"] = df["hour"].isin([6, 7, 8, 9]).astype(int)
    df["is_evening_peak"] = df["hour"].isin([17, 18, 19, 20]).astype(int)

    return df


def feature_engineering(run_id: str, is_future: bool = False):
    mode = "FUTUR" if is_future else "TRAIN"
    folder = "prediction" if is_future else "train"
    prefix = "imputed"

    print(f"=== Feature Engineering {mode} - Run ID: {run_id} ===\n")

    for side in ["arrivals", "departures"]:
        print(f"→ {side.capitalize()}...")
        bytes_data = load_from_s3(f"processed/{folder}/{run_id}", f"{prefix}_{side}_{run_id}.parquet")
        df = pd.read_parquet(io.BytesIO(bytes_data))
        
        df = enrich_aircraft_features(df)
        df = add_holiday_features(df)
        df = add_temporal_features(df)

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, compression="gzip")
        save_to_s3(buffer.getvalue(), f"processed/{folder}/{run_id}", f"final_{side}_{run_id}.parquet")
        print(f"   ✓ {side.capitalize()} : {len(df)} lignes\n")

    print(f"=== Feature Engineering {mode} terminé ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--is_future", action="store_true")
    args = parser.parse_args()
    
    feature_engineering(args.run_id, args.is_future)


if __name__ == "__main__":
    main()