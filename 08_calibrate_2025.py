"""Temel modeli 2025 gerçekleşmesine karşı kalibre et / doğrula."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import importlib; F = importlib.import_module("07_fundamental")

D = pd.read_parquet("data/tr_hourly.parquet")
act = D.loc["2025", "ptf_usd"]
actual = dict(avg=act.mean(), hour=act.groupby(act.index.hour).mean().values,
              month=act.groupby(act.index.month).mean().values, zero=int((act <= 1).sum()))

BASE_2025 = json.load(open("scripts/params_2025.json"))

def score(s):
    o = F.solve_year(s)
    hp = np.array(o["hour_profile"]); mp = np.array(o["month_avg"])
    return o, dict(avg_err=o["avg_price"] - actual["avg"],
                   hour_mae=float(np.abs(hp - actual["hour"]).mean()),
                   month_mae=float(np.abs(mp - actual["month"]).mean()),
                   hour_corr=float(np.corrcoef(hp, actual["hour"])[0, 1]),
                   month_corr=float(np.corrcoef(mp, actual["month"])[0, 1]))

if __name__ == "__main__":
    o, sc = score(BASE_2025)
    print(json.dumps({k: v for k, v in o.items() if k not in ("hour_profile",)}, indent=1))
    print("SKOR:", sc)
    print("gerçek saatlik:", np.round(actual["hour"], 1).tolist())
    print("model saatlik :", o["hour_profile"])
    print("gerçek aylık :", np.round(actual["month"], 1).tolist())
    print("model aylık  :", o["month_avg"])
    print("gerçek sıfır saat:", actual["zero"])
    json.dump(dict(model=o, score=sc, actual_hour=actual["hour"].round(2).tolist(),
                   actual_month=actual["month"].round(2).tolist(), actual_avg=actual["avg"],
                   actual_zero=actual["zero"]), open("out/calibration_2025.json", "w"), indent=1)
