import json, sys, itertools, numpy as np
sys.path.insert(0, "scripts")
exec(open("scripts/08_calibrate_2025.py").read().split('if __name__')[0])
base = dict(BASE_2025); base.update({"export_value": 40, "res_power_factor": 0.6})
grid = {"ror_share": [0.6, 0.75], "gas_min": [0.15, 0.25], "thermal_availability": [0.7, 0.8], "gas_usd_mwh_th": [35, 40], "lignite_min": [0.45, 0.65], "hydro_twh": [55, 65]}
rows = []
for vals in itertools.product(*grid.values()):
    s = dict(base); s.update(dict(zip(grid.keys(), vals)))
    o, sc = score(s)
    rows.append(dict(zip(grid.keys(), vals), **{k: round(float(v), 3) for k, v in sc.items()},
                     zero=o["zero_price_hours"], spread4=round(o["spread4_avg"], 1), cap_ratio=round(o["solar_capture_ratio"], 3)))
    print(rows[-1], flush=True)
json.dump(rows, open("out/calib_grid2.json", "w"), indent=1)
