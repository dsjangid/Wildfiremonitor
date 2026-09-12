# -*- coding: utf-8 -*-
"""
Production Server for Wildfire Response Prioritization System.
Provides RESTful APIs for:
- 9x9 Montesinho spatial grid risk mapping
- Constrained matroid portfolio selection (25 crews, max 4 per cell)
- What-if meteorological simulation
- Evaluation CSV dataset upload and inference
- One-click export of predictions.csv and selected_portfolio.csv
- Modular asset slot catalog
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

# Ensure parent directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import FeaturePipeline, PERMITTED_FEATURES
from src.model import HurdleRankEnsemble
from src.selection import select_response_portfolio
from src.metrics import (
    compute_ndcg_at_k,
    compute_spearman_correlation,
    compute_high_impact_recall,
    calculate_judging_rubric
)

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"),
    static_url_path="/static"
)

# Global in-memory state
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
TRAIN_CSV = os.path.join(DATA_DIR, "forestfires.csv")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# State container
STATE = {
    "pipeline": None,
    "model": None,
    "df_train": None,
    "df_eval_base": None,
    "df_eval_current": None,
    "eval_source_name": "forestfires.csv (default baseline)",
    "impact_scores": None,
    "selected_indices": None,
    "selected_df": None,
    "active_simulation": None
}


def initialize_model():
    """Load baseline data, fit pipeline strictly on train set, and generate baseline predictions."""
    print("Initializing Wildfire Response ML engine...")
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training dataset not found at: {TRAIN_CSV}")

    df_train = pd.read_csv(TRAIN_CSV)
    pipeline = FeaturePipeline().fit(df_train)
    X_train_scaled = pipeline.transform(df_train)

    model = HurdleRankEnsemble(random_state=42)
    model.fit(X_train_scaled, df_train['area'].values)

    STATE["df_train"] = df_train
    STATE["pipeline"] = pipeline
    STATE["model"] = model
    STATE["df_eval_base"] = df_train.copy()
    STATE["df_eval_current"] = df_train.copy()
    STATE["eval_source_name"] = "forestfires.csv (baseline)"
    STATE["active_simulation"] = None

    # Run baseline inference
    run_inference_and_selection(df_train.copy(), is_baseline=True)
    print("Wildfire Response ML engine ready.")


def run_inference_and_selection(df_eval: pd.DataFrame, is_baseline=False):
    """Run pipeline transform, ensemble predict, and constrained matroid selection."""
    pipeline = STATE["pipeline"]
    model = STATE["model"]

    # Transform evaluation features
    X_eval_scaled = pipeline.transform(df_eval)
    impact_scores = model.predict(X_eval_scaled)

    # Constrained selection (25 observations, max 4 per cell)
    portfolio_size = min(25, len(df_eval))
    selected_indices, selected_df = select_response_portfolio(
        df_eval=df_eval,
        impact_scores=impact_scores,
        portfolio_size=portfolio_size,
        max_per_cell=4
    )

    STATE["df_eval_current"] = df_eval
    STATE["impact_scores"] = impact_scores
    STATE["selected_indices"] = selected_indices
    STATE["selected_df"] = selected_df

    # Export to outputs directory
    pred_df = df_eval[['X', 'Y']].copy()
    pred_df.insert(0, 'row_id', np.arange(len(df_eval)))
    pred_df['impact_score'] = impact_scores
    pred_df['predicted_rank'] = pd.Series(impact_scores).rank(ascending=False, method='min').astype(int)

    pred_out_path = os.path.join(OUTPUTS_DIR, "predictions.csv")
    pred_df.to_csv(pred_out_path, index=False)

    portfolio_out_path = os.path.join(OUTPUTS_DIR, "selected_portfolio.csv")
    selected_df.to_csv(portfolio_out_path, index=False)

    return impact_scores, selected_df


def build_grid_response_payload():
    """Aggregate evaluation rows and selected crews across the 9x9 Montesinho grid."""
    df_eval = STATE["df_eval_current"]
    scores = STATE["impact_scores"]
    selected_df = STATE["selected_df"]

    # Prepare 9x9 grid cells: X from 1 to 9, Y from 1 to 9
    grid_cells = {}
    for x in range(1, 10):
        for y in range(1, 10):
            grid_cells[(x, y)] = {
                "x": x,
                "y": y,
                "observation_count": 0,
                "avg_impact": 0.0,
                "max_impact": 0.0,
                "crews_assigned": 0,
                "crew_details": [],
                "sample_weather": None
            }

    # Aggregate evaluation observations
    eval_temp = df_eval.copy()
    eval_temp['score'] = scores
    eval_temp['orig_idx'] = np.arange(len(df_eval))

    for idx, row in eval_temp.iterrows():
        cx = int(row['X'])
        cy = int(row['Y'])
        if 1 <= cx <= 9 and 1 <= cy <= 9:
            cell = grid_cells[(cx, cy)]
            cell["observation_count"] += 1
            score = float(row['score'])
            if score > cell["max_impact"]:
                cell["max_impact"] = score
                cell["sample_weather"] = {
                    "temp": float(row.get('temp', 0)),
                    "RH": float(row.get('RH', 0)),
                    "wind": float(row.get('wind', 0)),
                    "rain": float(row.get('rain', 0)),
                    "FFMC": float(row.get('FFMC', 0)),
                    "DMC": float(row.get('DMC', 0)),
                    "DC": float(row.get('DC', 0)),
                    "ISI": float(row.get('ISI', 0)),
                    "month": str(row.get('month', '')),
                    "day": str(row.get('day', ''))
                }

    # Compute cell averages
    grouped = eval_temp.groupby(['X', 'Y'])['score'].mean().to_dict()
    for (cx, cy), avg_sc in grouped.items():
        if (int(cx), int(cy)) in grid_cells:
            grid_cells[(int(cx), int(cy))]["avg_impact"] = float(avg_sc)

    # Add crew assignments from selected portfolio
    for _, crew in selected_df.iterrows():
        cx = int(crew['X'])
        cy = int(crew['Y'])
        if (cx, cy) in grid_cells:
            cell = grid_cells[(cx, cy)]
            cell["crews_assigned"] += 1
            cell["crew_details"].append({
                "priority_rank": int(crew['priority_rank']),
                "impact_score": round(float(crew['impact_score']), 4),
                "temp": float(crew.get('temp', 0)),
                "wind": float(crew.get('wind', 0)),
                "RH": float(crew.get('RH', 0)),
                "rain": float(crew.get('rain', 0)),
                "FFMC": float(crew.get('FFMC', 0)),
                "DMC": float(crew.get('DMC', 0)),
                "DC": float(crew.get('DC', 0)),
                "ISI": float(crew.get('ISI', 0)),
                "month": str(crew.get('month', '')),
                "day": str(crew.get('day', ''))
            })

    # Convert to list
    grid_list = list(grid_cells.values())

    # Selected crews list (ordered by priority)
    crew_list = []
    for _, crew in selected_df.sort_values(by='priority_rank').iterrows():
        crew_list.append({
            "priority_rank": int(crew['priority_rank']),
            "x": int(crew['X']),
            "y": int(crew['Y']),
            "impact_score": round(float(crew['impact_score']), 4),
            "temp": round(float(crew.get('temp', 0)), 1),
            "wind": round(float(crew.get('wind', 0)), 1),
            "RH": round(float(crew.get('RH', 0)), 1),
            "rain": round(float(crew.get('rain', 0)), 2),
            "FFMC": round(float(crew.get('FFMC', 0)), 1),
            "DMC": round(float(crew.get('DMC', 0)), 1),
            "DC": round(float(crew.get('DC', 0)), 1),
            "ISI": round(float(crew.get('ISI', 0)), 1),
            "month": str(crew.get('month', '')).upper(),
            "day": str(crew.get('day', '')).upper()
        })

    # Compute validation statistics
    cell_counts = selected_df.groupby(['X', 'Y']).size()
    max_per_cell_observed = int(cell_counts.max()) if len(cell_counts) > 0 else 0
    unique_cells_deployed = int(len(cell_counts))

    # Calculate ground truth metrics if 'area' exists in eval
    metrics = {
        "has_ground_truth": 'area' in df_eval.columns,
        "total_eval_rows": len(df_eval),
        "total_crews_selected": len(selected_df),
        "target_crews": 25,
        "max_per_cell_constraint": 4,
        "max_per_cell_observed": max_per_cell_observed,
        "constraint_satisfied": max_per_cell_observed <= 4,
        "unique_cells_covered": unique_cells_deployed,
        "eval_source": STATE["eval_source_name"],
        "is_simulation_active": STATE["active_simulation"] is not None
    }

    if 'area' in df_eval.columns:
        areas = df_eval['area'].values
        train_areas = STATE["df_train"]['area'].values
        ndcg = compute_ndcg_at_k(areas, scores, k=25)
        spearman = compute_spearman_correlation(areas, scores)
        recall = compute_high_impact_recall(train_areas, areas, STATE["selected_indices"])
        rubric = calculate_judging_rubric(
            mean_ndcg=ndcg,
            worst_ndcg=ndcg,
            mean_spearman=spearman,
            mean_recall=recall
        )
        metrics.update({
            "ndcg_at_25": round(float(ndcg), 4),
            "spearman_corr": round(float(spearman), 4),
            "high_impact_recall": round(float(recall), 4),
            "rubric_total_score": round(float(rubric['total_score_100']), 2)
        })

    return {
        "grid": grid_list,
        "crews": crew_list,
        "metrics": metrics,
        "simulation": STATE["active_simulation"]
    }


# ==========================================
# Flask Routes
# ==========================================

@app.route("/")
def index():
    """Serve main interactive 3D dashboard."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/donezo")
@app.route("/dashboard")
def donezo_dashboard():
    """Serve replicated Donezo productivity and task dashboard."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/portfolio")
def portfolio_page():
    """Serve dedicated 25-crew response portfolio page."""
    return send_from_directory(app.static_folder, "portfolio.html")


@app.route("/matrix")
def matrix_page():
    """Serve dedicated 9x9 spatial matrix and heatmap page."""
    return send_from_directory(app.static_folder, "matrix.html")


@app.route("/simulation")
def simulation_page():
    """Serve dedicated what-if weather simulation sandbox page."""
    return send_from_directory(app.static_folder, "simulation.html")


@app.route("/analytics")
def analytics_page():
    """Serve dedicated ML model evaluation and benchmark analytics page."""
    return send_from_directory(app.static_folder, "analytics.html")


@app.route("/assets")
def assets_page():
    """Serve modular asset catalog and visual customization page."""
    return send_from_directory(app.static_folder, "assets.html")


@app.route("/legacy")
@app.route("/prev")
@app.route("/old")
def legacy_dashboard():
    """Serve the original dark cyberpunk tactical 3D dashboard for side-by-side comparison."""
    return send_from_directory(app.static_folder, "legacy_dashboard.html")



@app.route("/api/status", methods=["GET"])
def api_status():
    """Return model status, training statistics, and current configuration."""
    return jsonify({
        "status": "online",
        "model_architecture": "Multi-Task Hurdle & Ranking Ensemble",
        "components": [
            {"name": "L2-Regularized Ridge Regressor", "weight": 0.50, "target": "log1p(area)"},
            {"name": "Huber Regressor (Outlier-Robust)", "weight": 0.20, "target": "log1p(area)"},
            {"name": "Calibrated Logistic Classifier", "weight": 0.30, "target": "P(burn >= 80th percentile)"}
        ],
        "permitted_features": PERMITTED_FEATURES,
        "geographic_bounds": {"grid_size": "9x9", "X_range": [1, 9], "Y_range": [1, 9]},
        "rules": {
            "portfolio_size": 25,
            "max_per_cell": 4,
            "zero_leakage": True
        },
        "current_eval_source": STATE["eval_source_name"]
    })


@app.route("/api/grid", methods=["GET"])
def api_grid():
    """Return current spatial 9x9 grid data and crew allocations."""
    data = build_grid_response_payload()
    return jsonify(data)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accept an uploaded evaluation CSV.
    Validates permitted input columns, executes inference & portfolio selection,
    and returns updated spatial grid and response allocations.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.csv'):
        return jsonify({"error": "Only CSV files are supported."}), 400

    try:
        # Support multiple encodings: UTF-8 (with or without BOM), Latin-1, CP1252
        raw_bytes = file.read()
        try:
            content = raw_bytes.decode('utf-8-sig')
        except UnicodeDecodeError:
            try:
                content = raw_bytes.decode('latin-1')
            except UnicodeDecodeError:
                content = raw_bytes.decode('cp1252', errors='replace')

        try:
            df_new = pd.read_csv(io.StringIO(content))
        except Exception as pe:
            return jsonify({"error": f"Malformed CSV file: {str(pe)}"}), 400

        # Strip whitespace from column names to handle comma-space separated CSV headers
        df_new.columns = df_new.columns.astype(str).str.strip()

        # Validate permitted features
        missing = [col for col in PERMITTED_FEATURES if col not in df_new.columns]
        if missing:
            return jsonify({
                "error": f"Uploaded CSV is missing required features: {missing}",
                "permitted_features": PERMITTED_FEATURES
            }), 400

        if len(df_new) < 25:
            return jsonify({
                "error": f"Evaluation dataset must contain at least 25 candidate rows (got {len(df_new)})."
            }), 400

        # Validate and coerce numeric columns, rejecting corrupted data with HTTP 400
        numeric_cols = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
        for col in numeric_cols:
            coerced = pd.to_numeric(df_new[col], errors='coerce')
            if coerced.isna().any():
                nan_count = int(coerced.isna().sum())
                return jsonify({
                    "error": f"Column '{col}' contains {nan_count} non-numeric or missing (NaN) values."
                }), 400
            df_new[col] = coerced

        # Validate Montesinho spatial coordinates (X in 1..9, Y in 1..9)
        if not df_new['X'].between(1, 9).all() or not df_new['Y'].between(1, 9).all():
            return jsonify({
                "error": "Montesinho spatial coordinates X and Y must be integers between 1 and 9."
            }), 400

        df_new['X'] = df_new['X'].round().astype(int)
        df_new['Y'] = df_new['Y'].round().astype(int)

        # Store pristine base dataframe for simulation branching
        STATE["df_eval_base"] = df_new.copy()
        STATE["df_eval_current"] = df_new.copy()
        STATE["eval_source_name"] = f"Uploaded: {filename} ({len(df_new)} rows)"
        STATE["active_simulation"] = None
        run_inference_and_selection(df_new)

        payload = build_grid_response_payload()
        payload["message"] = f"Successfully uploaded {filename} and executed constrained response selection."
        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": f"Failed to process CSV file: {str(e)}"}), 400


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Simulate what-if fire weather regimes across the evaluation dataset.
    Supports adjusting temp, RH, wind, rain, FFMC, DMC, DC, ISI, or month/day.
    Always branches from the pristine evaluation base dataset to prevent compounding mutation.
    """
    params = request.get_json(force=True) or {}

    try:
        # Base dataframe: always branch from the pristine evaluation base dataset
        if STATE.get("df_eval_base") is None:
            STATE["df_eval_base"] = STATE["df_train"].copy()

        base_df = STATE["df_eval_base"].copy()

        # Extract simulation adjustments
        temp_delta = float(params.get("temp_delta", 0.0))
        rh_delta = float(params.get("rh_delta", 0.0))
        wind_delta = float(params.get("wind_delta", 0.0))
        rain_override = params.get("rain_override", None)
        ffmc_override = params.get("ffmc_override", None)
        isi_override = params.get("isi_override", None)
        scenario_preset = params.get("scenario_preset", "Custom What-If")

        sim_df = base_df.copy()

        # Apply meteorological shifts
        if temp_delta != 0.0:
            sim_df['temp'] = np.clip(sim_df['temp'] + temp_delta, 0.0, 50.0)

        if rh_delta != 0.0:
            sim_df['RH'] = np.clip(sim_df['RH'] + rh_delta, 5.0, 100.0)

        if wind_delta != 0.0:
            sim_df['wind'] = np.clip(sim_df['wind'] + wind_delta, 0.0, 40.0)

        if rain_override is not None and rain_override != "":
            sim_df['rain'] = np.clip(float(rain_override), 0.0, 50.0)

        if ffmc_override is not None and ffmc_override != "":
            sim_df['FFMC'] = np.clip(float(ffmc_override), 15.0, 100.0)

        if isi_override is not None and isi_override != "":
            sim_df['ISI'] = np.clip(float(isi_override), 0.0, 60.0)

        is_shifted = (
            temp_delta != 0.0 or rh_delta != 0.0 or wind_delta != 0.0 or
            bool(rain_override) or bool(ffmc_override) or bool(isi_override)
        )
        if is_shifted:
            STATE["active_simulation"] = {
                "preset": scenario_preset,
                "temp_delta": temp_delta,
                "rh_delta": rh_delta,
                "wind_delta": wind_delta,
                "rain_override": rain_override,
                "ffmc_override": ffmc_override,
                "isi_override": isi_override
            }
        else:
            STATE["active_simulation"] = None

        run_inference_and_selection(sim_df)

        payload = build_grid_response_payload()
        payload["message"] = f"What-If Simulation applied: {scenario_preset}."
        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset evaluation data and what-if simulations back to baseline."""
    STATE["active_simulation"] = None
    STATE["eval_source_name"] = "forestfires.csv (baseline)"
    STATE["df_eval_base"] = STATE["df_train"].copy()
    STATE["df_eval_current"] = STATE["df_train"].copy()
    run_inference_and_selection(STATE["df_train"].copy(), is_baseline=True)
    payload = build_grid_response_payload()
    payload["message"] = "Reset system back to baseline forestfires.csv evaluation."
    return jsonify(payload)


@app.route("/api/download/predictions", methods=["GET"])
def download_predictions():
    """Download predictions.csv."""
    path = os.path.join(OUTPUTS_DIR, "predictions.csv")
    if not os.path.exists(path):
        return jsonify({"error": "predictions.csv not found"}), 404
    return send_file(
        path,
        as_attachment=True,
        download_name="predictions.csv",
        mimetype="text/csv"
    )


@app.route("/api/download/portfolio", methods=["GET"])
def download_portfolio():
    """Download selected_portfolio.csv."""
    path = os.path.join(OUTPUTS_DIR, "selected_portfolio.csv")
    if not os.path.exists(path):
        return jsonify({"error": "selected_portfolio.csv not found"}), 404
    return send_file(
        path,
        as_attachment=True,
        download_name="selected_portfolio.csv",
        mimetype="text/csv"
    )


@app.route("/api/assets", methods=["GET"])
def list_assets():
    """List available modular asset slots and their status, filtering out non-media documentation files."""
    assets_dir = os.path.join(app.static_folder, "assets")
    slots = []
    MEDIA_EXTENSIONS = {'.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.webm', '.json'}
    if os.path.exists(assets_dir):
        for root, _, files in os.walk(assets_dir):
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in MEDIA_EXTENSIONS:
                    rel = os.path.relpath(os.path.join(root, f), assets_dir).replace("\\", "/")
                    slots.append({
                        "slot_path": f"/static/assets/{rel}",
                        "filename": f,
                        "type": "custom" if "custom" in rel else "default",
                        "category": os.path.basename(root)
                    })
    return jsonify({
        "assets_directory": assets_dir,
        "available_slots": slots,
        "instructions": "Drop custom animations (.svg, .json, .webm, .gif) into web/static/assets/animations/ or images into web/static/assets/custom_images/ to customize visual themes."
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wildfire Response 3D Command Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)), help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    initialize_model()
    print(f"Starting Wildfire Response 3D Command Server on http://localhost:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False)
else:
    # When imported or run with WSGI
    initialize_model()
