"""Kıtlık/teklif katmanı parametrelerini (a,b,γ) 2025 saatlik fiyatlarına uydur."""
import json, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, "scripts")
import importlib; F = importlib.import_module("07_fundamental")
D = pd.read_parquet("data/tr_hourly.parquet"); act = D.loc["2025", "ptf_usd"].to_numpy()
s = json.load(open("scripts/params_2025.json"))
s.update({"export_value": 40, "res_power_factor": 0.6, "ror_share": 0.75, "gas_min": 0.15,
          "thermal_availability": 0.8, "gas_usd_mwh_th": 35, "lignite_min": 0.45, "hydro_twh": 55})
o = F.solve_year(s); u = o["_u"]; plp = o["_price_lp"]
def loss(p):
    a, b, g = p
    pr = np.clip(plp * (a + b * np.clip(u, 0, 1) ** g), 0, s["price_cap"])
    return np.abs(pr - act).mean()
best = minimize(loss, [0.5, 0.8, 2.0], method="Nelder-Mead", bounds=None, options=dict(maxiter=600))
a, b, g = best.x; print("a,b,g =", best.x.round(3), "MAE =", round(best.fun, 2), "LP MAE =", round(np.abs(plp - act).mean(), 2))
s["markup"] = dict(a=float(a), b=float(b), g=float(g))
json.dump(s, open("scripts/params_2025_calibrated.json", "w"), indent=1)
o2 = F.solve_year(s)
ah = pd.Series(act).groupby(D.loc["2025"].index.hour).mean().round(1).tolist()
print("gerçek saat:", ah); print("model saat :", o2["hour_profile"])
print({k: round(o2[k], 3) for k in ["avg_price", "solar_capture_ratio", "wind_capture_ratio", "zero_price_hours", "spread4_avg", "gas_twh", "coal_twh", "hydro_twh", "curtail_twh"]})
