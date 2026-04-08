
# HTTP entrypoint: train_llm_http

import os, io, json, logging, traceback, re
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence, permutation_importance

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "preds_llm")            # e.g., "structured/preds_llm"
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")      # split by local day
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

def _clean_numeric(s: pd.Series) -> pd.Series:
    # Strip $, commas, spaces; keep digits and dot
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _preprocess_features(df):
    # VIN truncation so that we can extract useful information rather than just a serial number
    if "vin" in df.columns:
        df["vin_prefix"] = df["vin"].astype(str).str[:10]
    else:
        df["vin_prefix"] = "unknown"

    # Retailer binary: 1 if "owner", 0 otherwise
    if "retailer" in df.columns:
        df["is_owner"] = df["retailer"].fillna("").str.lower().str.contains("for sale by owner").astype(int)
    else:
        df["is_owner"] = 0

    # Ensure sunroof and other categories are clean strings
    for col in ["sunroof", "transmission", "drivetrain", "color", "title_status", "make", "model"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
        else:
            df[col] = "unknown"
    return df


def generate_model_artifacts(model, X, y, cat_cols):
    artifacts = {}
    
    # Permutation Importance (Fast on smaller datasets)
    perm_importance = permutation_importance(model, X, y, n_repeats=3, random_state=42)
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean
    }).sort_values(by='importance', ascending=False)
    artifacts['importance_csv'] = importance_df.to_csv(index=False)
    
    # PDPs for Top 3 Features
    top_3 = importance_df['feature'].head(3).tolist()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, feat in enumerate(top_3):
        pd_results = partial_dependence(model, X, features=[feat], kind="average")
        ax[i].plot(pd_results['values'][0], pd_results['average'][0])
        ax[i].set_title(f'PDP: {feat}')
        ax[i].grid(True)
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    artifacts['pdp_png'] = buf.getvalue()
    plt.close(fig)
    
    return artifacts, importance_df

def _write_string_to_gcs(client: storage.Client, bucket: str, key: str, content: str):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(content, content_type="text/csv")

def _write_bytes_to_gcs(client: storage.Client, bucket: str, key: str, content: bytes):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(content, content_type="image/png")

def run_once(dry_run: bool = False, n_trials: int = 10, iterations: int = 500):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    df = _preprocess_features(df)

    cat_cols = ["make", "model", "transmission", "drivetrain", "color", 
                "sunroof", "vin_prefix", "title_status"]
    num_cols = ["year_num", "mileage_num", "is_owner"]
    feats = cat_cols + num_cols
    
    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numerics BEFORE counting/dropping ---
    orig_rows = len(df)
    df["price_num"]   = _clean_numeric(df["price"])
    df["year_num"]    = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    # Adding code to try to predict log(price) instead of raw price in order to account for price variation
    df["target_log"] = np.log1p(df["price_num"])
    target = "target_log"
    df = df.dropna(subset=[target])

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | with valid numeric price=%d", orig_rows, valid_price_rows)

    counts = df["date_local"].value_counts().sort_index()
    logging.info("Recent date counts (local): %s", json.dumps({str(k): int(v) for k, v in counts.tail(8).items()}))

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two distinct dates", "dates": [str(d) for d in unique_dates]}

    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    today_local = unique_dates[-1]
    train_df   = df[df["date_local"] <  today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    # --- Hyperparameter tuning with Optuna
    X_train = train_df[feats].copy()
    y_train = train_df[target]

    def objective(trial):
        param = {
            "iterations": 500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_strength": trial.suggest_float("random_strength", 1, 10),
            "loss_function": "RMSE",
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": 42
        }
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train, cat_features=cat_cols)
        scores = model.get_best_score()
        return scores.get('learn', {}).get('RMSE', 999999)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

# ---  Final Training with Best Params ---

    final_model = CatBoostRegressor(
        **study.best_params,
        iterations=500,
        verbose=False,
        allow_writing_files=False
    )
    final_model.fit(X_train, y_train, cat_features=cat_cols)

    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    out_key = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}/preds_llm.csv"


    if not dry_run:
        artifacts, imp_df = generate_model_artifacts(final_model, X_train, y_train, cat_cols)
        
        # Save Importance CSV
        imp_key = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}/importance.csv"
        _write_string_to_gcs(client, GCS_BUCKET, imp_key, artifacts['importance_csv'])
        
        # Save PDP Plot
        pdp_key = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}/pdp_top3.png"
        _write_bytes_to_gcs(client, GCS_BUCKET, pdp_key, artifacts['pdp_png'])

    # ---- Predict/evaluate on today's holdout (now includes actual price fields) ----
    mae_today = None
    preds_df = pd.DataFrame()
    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat_log = final_model.predict(holdout_df[feats])
        y_hat_dollars = np.expm1(y_hat_log)

        cols = ["scraped_at", "price", "year", "make", "model", "mileage", 
                    "transmission", "drivetrain", "color", "sunroof", 
                    "retailer", "vin", "title_status"]
        preds_df = holdout_df[cols].copy()
        preds_df["actual_price"] = holdout_df["price_num"]       # cleaned numeric truth
        preds_df["pred_price"]   = np.round(y_hat_dollars, 2)
        preds_df["actual_log_price"] = holdout_df["target_log"]       # cleaned numeric truth
        preds_df["pred_log_price"]   = np.round(y_hat_log, 4)

        if holdout_df["price_num"].notna().any():
            y_true = holdout_df["price_num"]
            mask = y_true.notna()
            if mask.any():
                mae_today = float(mean_absolute_error(y_true[mask], y_hat_dollars[mask]))

    # --- Output path: HOURLY folder structure ---

    if not dry_run and len(preds_df) > 0:
        _write_csv_to_gcs(client, GCS_BUCKET, out_key, preds_df)
        logging.info("Wrote predictions to gs://%s/%s (%d rows)", GCS_BUCKET, out_key, len(preds_df))
    else:
        logging.info("Dry run or no holdout rows; skip write. Would write to gs://%s/%s", GCS_BUCKET, out_key)

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_price_rows": valid_price_rows,
        "mae_today": mae_today,
        "output_key": out_key,
        "dry_run": dry_run,
        "timezone": TIMEZONE,
    }

def train_llm_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False)),
            n_trials=int(body.get("n_trials", 10)),
            iterations=int(body.get("iterations", 500))
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
