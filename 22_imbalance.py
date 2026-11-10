"""Gerçek SMF/PTF ile dengesizlik birim maliyetleri (TL ve USD/MWh).
2016-2025: poz = min(PTF,SMF)·0.97 ; neg = max(PTF,SMF)·1.03
2026+    : RG 29.12.2025 (DUY m.110): neg = max(V,PTF,SMF)·(1+k); poz = min(PTF,SMF)·(1−l),
           min<V ise poz = −B·(1+l). V=150, B=100 TL. k,l: sistem yönüyle aynı taraf %6, ters taraf %3
           (resmi Kurul kararı bulunamadı; eptr2 kütüphanesinin uyguladığı değerler).
Not: 2026 kuralı 15 dakikalık SMF uçlarını kullanır; elimizde saatlik SMF var → maliyet alt sınır."""
import pandas as pd, numpy as np, json
E = pd.read_parquet("data/epias_hourly.parquet")
ptf, smf = E.ptf_tl.astype(float), E.smf_tl.astype(float)
fx = (E.ptf_tl / E.ptf_usd_e).where(E.ptf_usd_e > 0.5).ffill().bfill()
deficit = E.direction_id == 1; surplus = E.direction_id == 3
new = E.index >= "2026-01-01"
k = np.where(new, np.where(deficit, 0.06, 0.03), 0.03)
l = np.where(new, np.where(surplus, 0.06, 0.03), 0.03)
mn, mx = np.minimum(ptf, smf), np.maximum(ptf, smf)
neg = np.where(new, np.maximum(mx, 150) * (1 + k), mx * 1.03)
pos = np.where(new & (mn < 150), -100 * (1 + l), mn * (1 - l))
E["up_cost_tl"] = ptf - pos          # fazla üretim: PTF yerine poz fiyat
E["down_cost_tl"] = neg - ptf        # eksik üretim: açığı neg fiyattan alır
E["up_cost_usd"] = E.up_cost_tl / fx; E["down_cost_usd"] = E.down_cost_tl / fx
E["smf_usd"] = smf / fx
E[["up_cost_tl", "down_cost_tl", "up_cost_usd", "down_cost_usd", "smf_usd", "direction_id"]].to_parquet("data/imbalance_unit_costs.parquet")
y = E.groupby(E.index.year)
out = pd.DataFrame({
    "smf_usd": y.smf_usd.mean(), "ptf_usd": y.ptf_usd_e.mean(),
    "abs_gap_usd": (E.smf_usd - E.ptf_usd_e).abs().groupby(E.index.year).mean(),
    "deficit_share": y.apply(lambda g: (g.direction_id == 1).mean()),
    "surplus_share": y.apply(lambda g: (g.direction_id == 3).mean()),
    "up_cost_usd": y.up_cost_usd.mean(), "down_cost_usd": y.down_cost_usd.mean(),
    "up_cost_pct": y.up_cost_tl.sum() / y.ptf_tl.sum(), "down_cost_pct": y.down_cost_tl.sum() / y.ptf_tl.sum()}).round(4)
print(out.to_string())
# 2026 öncesi/sonrası (aynı 9 ay)
for lab, a, b in [("2025 Oca-Eki", "2025-01-01", "2025-10-28"), ("2026 Oca-Eki", "2026-01-01", "2026-10-28")]:
    g = E.loc[a:b]; print(lab, "up", round(g.up_cost_usd.mean(), 2), "down", round(g.down_cost_usd.mean(), 2), "up%", round(g.up_cost_tl.sum() / g.ptf_tl.sum(), 4), "down%", round(g.down_cost_tl.sum() / g.ptf_tl.sum(), 4))
out.index = out.index.astype(str); out.index.name = "year"; out.to_csv("out/imbalance_annual.csv")
