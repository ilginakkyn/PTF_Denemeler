"""Kantil bandının kalibrasyonu (walk-forward konformal genişletme).
Her test ayı için, YALNIZ önceki 3 ayın gerçekleşen hatalarına bakarak bandı genişletme katsayısı seçilir."""
import pandas as pd, numpy as np, json
b = pd.read_parquet("out/ptf_backtest.parquet")
mid = b["lgb_q50"]; lo = (mid - b["lgb_q10"]).clip(lower=0); hi = (b["lgb_q90"] - mid).clip(lower=0)
months = b.index.to_period("M").unique()
new_lo = pd.Series(np.nan, b.index); new_hi = pd.Series(np.nan, b.index)
for i, m in enumerate(months):
    mk = b.index.to_period("M") == m
    if i < 3:
        k = 1.0
    else:
        past = b.index.to_period("M").isin(months[i-3:i])
        ks = np.linspace(0.8, 3.0, 45)
        cov = [(((b.actual >= mid - kk*lo) & (b.actual <= mid + kk*hi))[past]).mean() for kk in ks]
        k = ks[np.argmin(np.abs(np.array(cov) - 0.80))]
    new_lo[mk] = (mid - k*lo)[mk].clip(lower=0); new_hi[mk] = (mid + k*hi)[mk]
b["q10_cal"], b["q90_cal"] = new_lo, new_hi
cov_raw = ((b.actual >= b.lgb_q10) & (b.actual <= b.lgb_q90)).mean()
ev = b.index >= months[3].to_timestamp()
cov_cal = ((b.actual >= b.q10_cal) & (b.actual <= b.q90_cal))[ev].mean()
print("ham kapsama", round(cov_raw, 3), "kalibre (ilk 3 ay hariç)", round(cov_cal, 3))
b.to_parquet("out/ptf_backtest.parquet")
r = json.load(open("out/ptf_eval.json")); r["q10_q90_coverage_calibrated"] = round(float(cov_cal), 3)
r["band_width_raw"] = round(float((b.lgb_q90 - b.lgb_q10).mean()), 2); r["band_width_cal"] = round(float((b.q90_cal - b.q10_cal)[ev].mean()), 2)
json.dump(r, open("out/ptf_eval.json", "w"), indent=1, default=str)
print(r["band_width_raw"], r["band_width_cal"])
