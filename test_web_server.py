# -*- coding: utf-8 -*-
"""
Automated Verification Test Suite for Wildfire Response Web Server & Endpoints.
Tests:
- API status and model configuration
- 9x9 Montesinho spatial grid payload and 25-crew matroid selection
- Cell quota constraint enforcement (max 4 per cell)
- What-if fire weather simulation (extreme values, clipping, re-dispatch)
- CSV upload endpoint (valid CSV, missing columns rejection, small dataset rejection)
- Download endpoints (predictions.csv and selected_portfolio.csv)
- Modular asset slot discovery
- Reset baseline functionality
"""

import os
import sys
import io
import json
import unittest
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.server import app, initialize_model, STATE, OUTPUTS_DIR


class TestWildfireWebServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        initialize_model()
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_index_page(self):
        """Verify root page serves HTML with 3D Command Center title and scripts."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        response.close()
        self.assertIn("Montesinho Wildfire Response 3D Command Center", html)
        self.assertIn("webgl-canvas", html)
        self.assertIn("/static/js/app.js", html)
        self.assertIn("predictions.csv", html)
        self.assertIn("portfolio.csv", html)

    def test_02_api_status(self):
        """Verify API status endpoint returns architecture and permitted features."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["status"], "online")
        self.assertIn("Hurdle", data["model_architecture"])
        self.assertEqual(len(data["components"]), 3)
        self.assertEqual(data["rules"]["portfolio_size"], 25)
        self.assertEqual(data["rules"]["max_per_cell"], 4)
        self.assertEqual(len(data["permitted_features"]), 12)

    def test_03_api_grid_structure_and_constraints(self):
        """Verify 9x9 grid has exactly 81 cells, 25 crews selected, and max per cell <= 4."""
        response = self.client.get("/api/grid")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))

        self.assertIn("grid", data)
        self.assertIn("crews", data)
        self.assertIn("metrics", data)

        # 9x9 = 81 cells
        self.assertEqual(len(data["grid"]), 81)

        # Verify coordinates range 1..9
        for cell in data["grid"]:
            self.assertTrue(1 <= cell["x"] <= 9)
            self.assertTrue(1 <= cell["y"] <= 9)
            self.assertTrue(0 <= cell["crews_assigned"] <= 4)

        # Verify portfolio crews
        self.assertEqual(len(data["crews"]), 25)
        # Check priority ranks 1 to 25
        ranks = [c["priority_rank"] for c in data["crews"]]
        self.assertEqual(ranks, list(range(1, 26)))

        # Verify strict quota <= 4 per cell
        cell_tally = {}
        for c in data["crews"]:
            key = (c["x"], c["y"])
            cell_tally[key] = cell_tally.get(key, 0) + 1

        max_in_cell = max(cell_tally.values())
        self.assertLessEqual(max_in_cell, 4, f"Observed {max_in_cell} crews in a single cell, exceeding quota 4!")
        self.assertTrue(data["metrics"]["constraint_satisfied"])
        self.assertEqual(data["metrics"]["max_per_cell_observed"], max_in_cell)

    def test_04_what_if_simulation(self):
        """Verify what-if simulation modifies fire conditions and re-allocates crews."""
        sim_payload = {
            "scenario_preset": "Severe Summer Heatwave & High Wind",
            "temp_delta": 12.0,
            "rh_delta": -25.0,
            "wind_delta": 10.0,
            "ffmc_override": 95.0,
            "isi_override": 35.0
        }
        response = self.client.post("/api/simulate", json=sim_payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(len(data["crews"]), 25)
        self.assertTrue(data["metrics"]["constraint_satisfied"])
        self.assertLessEqual(data["metrics"]["max_per_cell_observed"], 4)
        self.assertEqual(data["simulation"]["preset"], "Severe Summer Heatwave & High Wind")

    def test_05_upload_valid_csv(self):
        """Verify upload endpoint handles valid evaluation CSV and updates outputs."""
        sample_path = os.path.join(PROJECT_ROOT, "data", "forestfires.csv")
        with open(sample_path, "rb") as f:
            csv_bytes = f.read()

        data = {
            'file': (io.BytesIO(csv_bytes), 'eval_test.csv')
        }
        response = self.client.post(
            "/api/upload",
            data=data,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        resp_json = json.loads(response.get_data(as_text=True))
        self.assertIn("eval_test.csv", resp_json["message"])
        self.assertEqual(len(resp_json["crews"]), 25)
        self.assertLessEqual(resp_json["metrics"]["max_per_cell_observed"], 4)

    def test_06_upload_invalid_csv_rejected(self):
        """Verify upload rejects CSV missing required features or with insufficient rows."""
        # 1. Missing columns
        bad_df = pd.DataFrame({
            "X": [1, 2],
            "Y": [3, 4]
        })
        csv_buf = io.BytesIO(bad_df.to_csv(index=False).encode('utf-8'))
        response = self.client.post(
            "/api/upload",
            data={'file': (csv_buf, 'bad.csv')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn("missing required features", data["error"])

        # 2. Too few rows (< 25)
        df_full = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "forestfires.csv")).head(10)
        csv_buf2 = io.BytesIO(df_full.to_csv(index=False).encode('utf-8'))
        response2 = self.client.post(
            "/api/upload",
            data={'file': (csv_buf2, 'short.csv')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response2.status_code, 400)
        data2 = json.loads(response2.get_data(as_text=True))
        self.assertIn("at least 25 candidate rows", data2["error"])

    def test_07_download_endpoints(self):
        """Verify download endpoints return valid predictions.csv and selected_portfolio.csv."""
        # predictions.csv
        r_pred = self.client.get("/api/download/predictions")
        self.assertEqual(r_pred.status_code, 200)
        self.assertEqual(r_pred.mimetype, "text/csv")
        pred_df = pd.read_csv(io.BytesIO(r_pred.data))
        r_pred.close()
        self.assertIn("row_id", pred_df.columns)
        self.assertIn("impact_score", pred_df.columns)
        self.assertIn("predicted_rank", pred_df.columns)
        self.assertIn("X", pred_df.columns)
        self.assertIn("Y", pred_df.columns)

        # selected_portfolio.csv
        r_port = self.client.get("/api/download/portfolio")
        self.assertEqual(r_port.status_code, 200)
        self.assertEqual(r_port.mimetype, "text/csv")
        port_df = pd.read_csv(io.BytesIO(r_port.data))
        r_port.close()
        self.assertEqual(len(port_df), 25)
        self.assertIn("priority_rank", port_df.columns)
        self.assertIn("impact_score", port_df.columns)
        max_in_cell = port_df.groupby(["X", "Y"]).size().max()
        self.assertLessEqual(max_in_cell, 4)

    def test_08_modular_asset_slots(self):
        """Verify modular asset slots catalog lists all icons and animation slots."""
        response = self.client.get("/api/assets")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn("available_slots", data)
        slot_names = [s["filename"] for s in data["available_slots"]]
        self.assertIn("crew_marker.svg", slot_names)
        self.assertIn("flame_icon.svg", slot_names)
        self.assertIn("drone_icon.svg", slot_names)
        self.assertIn("shield_icon.svg", slot_names)
        self.assertIn("radar_sweep.svg", slot_names)
        self.assertIn("fire_pulse.svg", slot_names)

    def test_09_reset_baseline(self):
        """Verify reset endpoint restores original baseline state."""
        response = self.client.post("/api/reset")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIsNone(data["simulation"])
        self.assertEqual(len(data["crews"]), 25)
        self.assertTrue(data["metrics"]["constraint_satisfied"])


if __name__ == '__main__':
    unittest.main()
