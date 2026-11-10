"""2035 senaryoları: talep × arz × depolama (+ hidroloji ve karbon duyarlılığı).
Temel model 2025'e kalibre edilmiş parametrelerle (params_2025 + kalibrasyon) çalışır; 2035
kapasiteleri scripts/params_2035.json'dan gelir. Saatlik şekiller 2025 hava yılıdır.
"""
import json, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import importlib; F = importlib.import_module("07_fundamental")
cal = json.load(open("scripts/params_2025.json"))
cal.update(json.load(open("scripts/calibration_choice.json")))
P = json.load(open("scripts/params_2035.json"))

def scen(dem, sup, sto, hydro="ortalama", co2=0.0):
    s = dict(cal); s.pop("markup", None)
    s.update(P["common"])
    s.update(P["supply"][sup]); s.update(P["storage"][sto])
    d = P["demand"][dem]
    s["demand_twh"] = d["total_twh_epias_basis"] - d["ev_twh"]; s["ev_twh"] = d["ev_twh"]
    s["hydro_twh"] = P["hydro"][hydro]
    s["co2_usd_t"] = co2
    return s

rows = []
def run(label, **kw):
    o = F.solve_year(scen(**kw))
    r = dict(label=label, **kw)
    r.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if not k.startswith("_")})
    rows.append(r); print(label, {k: r[k] for k in ["avg_price", "solar_capture_ratio", "zero_price_hours", "curtail_twh", "curtail_share_vre", "gas_twh", "vre_share_of_demand", "spread4_avg"]}, flush=True)

for dem, sup, sto in itertools.product(["dusuk", "baz", "yuksek"], ["uep2022", "mevcut_hiz", "hedef_120"], ["yok", "uep_7.5GW_2s", "yuksek_20GW_4s"]):
    run(f"{dem}|{sup}|{sto}", dem=dem, sup=sup, sto=sto)
for hy in ["kurak", "yagisli"]:
    run(f"baz|hedef_120|uep_7.5GW_2s|{hy}", dem="baz", sup="hedef_120", sto="uep_7.5GW_2s", hydro=hy)
for co2 in [30.0, 60.0]:
    run(f"baz|hedef_120|uep_7.5GW_2s|co2={co2:.0f}", dem="baz", sup="hedef_120", sto="uep_7.5GW_2s", co2=co2)
# 2025 kalibrasyon yılı aynı modelle (karşılaştırma için)
s25 = dict(cal); s25["hydro_twh"] = 57.2; s25["wind_mw"] = 13800
o = F.solve_year(s25); r = dict(label="2025_model"); r.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if not k.startswith("_")}); rows.append(r)
json.dump(rows, open("out/scenarios_2035.json", "w"), indent=1)
