import json, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
exec(open("scripts/08_calibrate_2025.py").read().split('if __name__')[0])
base = dict(BASE_2025); base.update(json.load(open("scripts/calibration_choice.json")))
base.update({"hydro_twh": 57.2, "wind_mw": 13800})
# 2025 gerçekleşen (EPİAŞ gerçek zamanlı üretim): linyit+asfaltit+taşkömürü 45.3, ithal kömür 76.1, gaz 78.4
grid = {"lignite_availability": [0.5, 0.55, 0.65], "coal_availability": [0.8, 0.88], "gas_min": [0.15, 0.25],
        "ror_share": [0.6, 0.75], "gas_usd_mwh_th": [35.6], "lignite_min": [0.45, 0.6]}
rows = []
for vals in itertools.product(*grid.values()):
    s = dict(base); s.update(dict(zip(grid.keys(), vals)))
    o, sc = score(s)
    mixerr = abs(o["lignite_twh"] - 45.3) + abs(o["import_coal_twh"] - 76.1) + abs(o["gas_twh"] - 78.4)
    rows.append(dict(zip(grid.keys(), vals), **{k: round(float(v), 3) for k, v in sc.items()}, mixerr=round(mixerr, 1),
                     lig=round(o["lignite_twh"], 1), coal=round(o["import_coal_twh"], 1), gas=round(o["gas_twh"], 1),
                     zero=o["zero_price_hours"], spread4=round(o["spread4_avg"], 1), cap=round(o["solar_capture_ratio"], 3)))
    print(rows[-1], flush=True)
json.dump(rows, open("out/calib_grid3.json", "w"), indent=1)
