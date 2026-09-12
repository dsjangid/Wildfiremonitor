# -*- coding: utf-8 -*-
"""
Live Network Integration Test.
Launches the Flask web server in a subprocess on port 5055, performs live HTTP requests,
verifies API payloads, downloaded files, and shuts down cleanly.
"""

import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
import json
import pandas as pd
import io

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_PY = os.path.join(PROJECT_ROOT, "web", "server.py")
PORT = 5055
BASE_URL = f"http://127.0.0.1:{PORT}"


def run_live_test():
    print(f"Starting server subprocess on {BASE_URL}...")
    env = os.environ.copy()
    env["PORT"] = str(PORT)

    proc = subprocess.Popen(
        [sys.executable, SERVER_PY],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready
    ready = False
    for i in range(15):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/status", timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass

    if not ready:
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        print("Server failed to start. Stdout:", stdout)
        print("Stderr:", stderr)
        sys.exit(1)

    print("Server is UP and responsive!")

    try:
        # 1. Test GET /
        print("Checking GET /...")
        with urllib.request.urlopen(f"{BASE_URL}/") as resp:
            assert resp.status == 200
            html = resp.read().decode('utf-8')
            assert "Montesinho Wildfire Response 3D Command Center" in html
        print("  -> Index HTML served successfully.")

        # 2. Test GET /api/status
        print("Checking GET /api/status...")
        with urllib.request.urlopen(f"{BASE_URL}/api/status") as resp:
            assert resp.status == 200
            status_data = json.loads(resp.read().decode('utf-8'))
            assert status_data["status"] == "online"
            assert status_data["rules"]["portfolio_size"] == 25
            assert status_data["rules"]["max_per_cell"] == 4
        print("  -> Status endpoint verified.")

        # 3. Test GET /api/grid
        print("Checking GET /api/grid...")
        with urllib.request.urlopen(f"{BASE_URL}/api/grid") as resp:
            assert resp.status == 200
            grid_data = json.loads(resp.read().decode('utf-8'))
            assert len(grid_data["grid"]) == 81
            assert len(grid_data["crews"]) == 25
            assert grid_data["metrics"]["max_per_cell_observed"] <= 4
            assert grid_data["metrics"]["constraint_satisfied"] is True
        print(f"  -> Grid endpoint verified (81 cells, 25 crews, max in cell: {grid_data['metrics']['max_per_cell_observed']}).")

        # 4. Test POST /api/simulate
        print("Checking POST /api/simulate...")
        req = urllib.request.Request(
            f"{BASE_URL}/api/simulate",
            data=json.dumps({"temp_delta": 8, "rh_delta": -20, "wind_delta": 6, "scenario_preset": "Test Heatwave"}).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            sim_data = json.loads(resp.read().decode('utf-8'))
            assert len(sim_data["crews"]) == 25
            assert sim_data["metrics"]["max_per_cell_observed"] <= 4
        print("  -> What-if simulation endpoint verified.")

        # 5. Test Download endpoints
        print("Checking GET /api/download/predictions...")
        with urllib.request.urlopen(f"{BASE_URL}/api/download/predictions") as resp:
            assert resp.status == 200
            df_pred = pd.read_csv(io.BytesIO(resp.read()))
            assert len(df_pred) == 517
            assert "impact_score" in df_pred.columns
            assert "predicted_rank" in df_pred.columns
        print(f"  -> predictions.csv verified ({len(df_pred)} rows).")

        print("Checking GET /api/download/portfolio...")
        with urllib.request.urlopen(f"{BASE_URL}/api/download/portfolio") as resp:
            assert resp.status == 200
            df_port = pd.read_csv(io.BytesIO(resp.read()))
            assert len(df_port) == 25
            assert "priority_rank" in df_port.columns
            assert "impact_score" in df_port.columns
            max_c = df_port.groupby(["X", "Y"]).size().max()
            assert max_c <= 4
        print(f"  -> selected_portfolio.csv verified (25 rows, max per cell: {max_c} <= 4).")

        # 6. Test GET /api/assets
        print("Checking GET /api/assets...")
        with urllib.request.urlopen(f"{BASE_URL}/api/assets") as resp:
            assert resp.status == 200
            asset_data = json.loads(resp.read().decode('utf-8'))
            assert len(asset_data["available_slots"]) >= 6
        print(f"  -> Modular asset slots verified ({len(asset_data['available_slots'])} detected).")

        print("\nALL LIVE INTEGRATION CHECKS PASSED PERFECTLY!\n")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print("Server process cleanly shut down.")


if __name__ == "__main__":
    run_live_test()
