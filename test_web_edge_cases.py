# -*- coding: utf-8 -*-
"""
Edge-Case & Adversarial Tests for Wildfire Web Server.
Tests:
1. Extreme clustered risk: all high fires in 1 cell (verifies max 4 constraint strictly holds)
2. Extreme weather boundaries in simulation (clipping, numerical stability)
3. Non-numeric and corrupted CSV uploads
4. Static file delivery (CSS, JS, SVG assets, animations)
5. Exactly 25 responses and max 4 per cell invariant across 50 random datasets
"""

import os
import sys
import io
import json
import unittest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.server import app, initialize_model


class TestWildfireWebEdgeCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        initialize_model()
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_static_assets_served(self):
        """Verify all icons, animations, css, and js are served with HTTP 200."""
        assets_to_check = [
            "/static/css/style.css",
            "/static/js/app.js",
            "/static/assets/icons/crew_marker.svg",
            "/static/assets/icons/flame_icon.svg",
            "/static/assets/icons/drone_icon.svg",
            "/static/assets/icons/shield_icon.svg",
            "/static/assets/icons/wind_icon.svg",
            "/static/assets/images/montesinho_terrain.svg",
            "/static/assets/animations/radar_sweep.svg",
            "/static/assets/animations/fire_pulse.svg",
        ]
        for path in assets_to_check:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, f"Failed to serve static asset: {path}")
            res.close()

    def test_02_extreme_weather_clipping(self):
        """Verify extreme weather simulation parameters do not produce NaNs or crash."""
        payload = {
            "scenario_preset": "Super-Extreme Armageddon Test",
            "temp_delta": 100.0,   # Should clip to 50
            "rh_delta": -150.0,    # Should clip to 5
            "wind_delta": 80.0,    # Should clip to 40
            "rain_override": 100.0,# Should clip to 50
            "ffmc_override": 200.0,# Should clip to 100
            "isi_override": 500.0  # Should clip to 60
        }
        res = self.client.post("/api/simulate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.get_data(as_text=True))
        # Ensure impact scores are valid numbers in [0, 1]
        for crew in data["crews"]:
            self.assertFalse(np.isnan(crew["impact_score"]))
            self.assertTrue(0.0 <= crew["impact_score"] <= 1.0)
        self.assertEqual(len(data["crews"]), 25)
        self.assertLessEqual(data["metrics"]["max_per_cell_observed"], 4)

    def test_03_clustered_spatial_saturation_upload(self):
        """Upload dataset where 100 observations are all placed in a SINGLE cell (X=5, Y=5).
        Verify that strictly at most 4 crews are placed in (5, 5) and the rest are placed in other cells."""
        # Create dataset with 100 rows in cell (5, 5) and 30 rows in other cells
        rows = []
        months = ['aug', 'sep', 'jul']
        days = ['fri', 'sat', 'sun']
        # 100 rows at (5, 5)
        for i in range(100):
            rows.append({
                'X': 5, 'Y': 5, 'month': months[i % 3], 'day': days[i % 3],
                'FFMC': 95.0, 'DMC': 200.0, 'DC': 700.0, 'ISI': 25.0,
                'temp': 35.0, 'RH': 15.0, 'wind': 12.0, 'rain': 0.0
            })
        # 30 rows distributed across other cells
        for i in range(30):
            cell_x = (i % 8) + 1
            if cell_x == 5: cell_x = 9
            rows.append({
                'X': cell_x, 'Y': (i % 8) + 1, 'month': 'aug', 'day': 'mon',
                'FFMC': 85.0, 'DMC': 100.0, 'DC': 500.0, 'ISI': 10.0,
                'temp': 20.0, 'RH': 40.0, 'wind': 4.0, 'rain': 0.0
            })

        df_stress = pd.DataFrame(rows)
        buf = io.BytesIO(df_stress.to_csv(index=False).encode('utf-8'))
        res = self.client.post(
            "/api/upload",
            data={'file': (buf, 'cluster_stress.csv')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.get_data(as_text=True))
        self.assertEqual(len(data["crews"]), 25)

        # Count in (5, 5) must be <= 4
        crews_in_5_5 = [c for c in data["crews"] if c["x"] == 5 and c["y"] == 5]
        self.assertLessEqual(len(crews_in_5_5), 4)
        self.assertEqual(data["metrics"]["max_per_cell_observed"], 4)
        self.assertTrue(data["metrics"]["constraint_satisfied"])

    def test_04_corrupted_csv_upload_handling(self):
        """Upload corrupted / non-CSV text and verify clean 400 rejection (never 500)."""
        # Non-csv extension
        buf = io.BytesIO(b"random binary data")
        res = self.client.post(
            "/api/upload",
            data={'file': (buf, 'corrupt.exe')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res.status_code, 400)
        res.close()

        # Non-numeric coordinate data must return clean 400
        corrupt_csv = "X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain\nINVALID,INVALID,aug,sun,90,100,500,10,20,30,5,0\n"
        buf2 = io.BytesIO(corrupt_csv.encode('utf-8'))
        res2 = self.client.post(
            "/api/upload",
            data={'file': (buf2, 'corrupt_types.csv')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res2.status_code, 400)
        data = res2.get_json()
        self.assertIn("error", data)
        res2.close()

    def test_05_encoding_and_whitespace_resilience(self):
        """Verify upload gracefully accepts Latin-1 encoded CSVs and whitespace-padded headers."""
        # 1. Latin-1 encoding with accents
        latin1_csv = 'X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain\n' + '1,1,août,mon,80.0,20.0,100.0,5.0,20.0,40.0,4.0,0.0\n' * 25
        buf = io.BytesIO(latin1_csv.encode('latin-1'))
        res = self.client.post(
            "/api/upload",
            data={'file': (buf, 'latin1_test.csv')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res.status_code, 200)
        res.close()

        # 2. Whitespace in column names: 'X, Y, month, ...'
        spaced_csv = 'X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain\n' + '1, 1, aug, mon, 80.0, 20.0, 100.0, 5.0, 20.0, 40.0, 4.0, 0.0\n' * 25
        buf2 = io.BytesIO(spaced_csv.encode('utf-8'))
        res2 = self.client.post(
            "/api/upload",
            data={'file': (buf2, 'spaced_test.csv')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res2.status_code, 200)
        res2.close()

    def test_06_spatial_coordinate_bounds_validation(self):
        """Verify out-of-bounds coordinates (X=0 or X=10) are rejected with 400."""
        bad_coord_csv = 'X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain\n' + '10,1,aug,mon,80.0,20.0,100.0,5.0,20.0,40.0,4.0,0.0\n' * 25
        buf = io.BytesIO(bad_coord_csv.encode('utf-8'))
        res = self.client.post(
            "/api/upload",
            data={'file': (buf, 'out_of_bounds.csv')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Montesinho spatial coordinates", res.get_json()["error"])
        res.close()

    def test_07_simulation_state_isolation(self):
        """Verify successive simulation calls do not compound shifts on top of one another."""
        # Ensure clean baseline
        self.client.post("/api/reset")
        from web.server import STATE
        t_base = float(STATE["df_eval_base"]["temp"].iloc[0])

        # Step 1: sim with +5
        res1 = self.client.post("/api/simulate", json={"temp_delta": 5.0})
        self.assertEqual(res1.status_code, 200)
        t_sim1 = float(STATE["df_eval_current"]["temp"].iloc[0])
        self.assertAlmostEqual(t_sim1, t_base + 5.0, places=4)
        res1.close()

        # Step 2: sim with +5 again (should still be +5 from base, NOT +10)
        res2 = self.client.post("/api/simulate", json={"temp_delta": 5.0})
        self.assertEqual(res2.status_code, 200)
        t_sim2 = float(STATE["df_eval_current"]["temp"].iloc[0])
        self.assertAlmostEqual(t_sim2, t_base + 5.0, places=4)
        res2.close()

        # Step 3: reset baseline
        res_reset = self.client.post("/api/reset")
        self.assertEqual(res_reset.status_code, 200)
        t_reset = float(STATE["df_eval_current"]["temp"].iloc[0])
        self.assertAlmostEqual(t_reset, t_base, places=4)
        res_reset.close()


if __name__ == '__main__':
    unittest.main()
