"""Ampirik fiyat-şekil modeli: saatlik göreli fiyat = f(göreli artık yük, saat, ay).

Maliyet bazlı LP gün içi fiyat farkını gerçekte olduğundan düşük gösterdi (2025 kalibrasyonu:
model 4 saatlik fark ~$17-23, gerçek ~$40; GES yakalama oranı model 0.91, gerçek ~0.80-0.83).
Bu yüzden şekli gerçek veriden öğreniyoruz:
  r_t = PTF_t / (30 günlük ortalama PTF)          hedef
  z_t = (Tüketim_t − GES_t − RES_t) / 30 günlük ort. tüketim   ana girdi (artık yük)
Eğitim: 2024-07..2025-12 ; DIŞ-ÖRNEKLEM test: 2026-01..2026-08 (GES'in çok daha yüksek olduğu dönem).
Toplam GES saatlik = lisanslı GES şekli × toplam kurulu güç (aylık, ETKB) — 2025 toplamı 38.1 TWh'e
(üretimin %10.5'i) ölçeklendi.
"""
import json, numpy as np, pandas as pd, lightgbm as lgb
D = pd.read_parquet("data/tr_hourly.parquet").loc["2024-06-01":"2026-08-27 23:00"]
cap_pts = json.load(open("scripts/capacity_points.json"))          # {"YYYY-MM-DD": MW}
sc = pd.Series({pd.Timestamp(k): v for k, v in cap_pts["solar_total_mw"].items()}).sort_index()
solar_cap = sc.reindex(sc.index.union(D.index)).interpolate("time").reindex(D.index).ffill().bfill()
A = D["act_solar"].astype(float)
lic_cap = A.rolling(24 * 30, min_periods=24 * 7).max().bfill() / 0.85
cf = (A / lic_cap).clip(0, 1).fillna(0)
solar_raw = cf * solar_cap
k = 38.1e6 / solar_raw.loc["2025"].sum()
solar_tot = solar_raw * k
print(f"GES ölçek katsayısı k={k:.3f}  (2025 ima edilen kapasite faktörü: {solar_tot.loc['2025'].sum()/ (solar_cap.loc['2025'].mean()*8760):.3f})")
L = D["act_consumption"].astype(float).fillna(D["consumption_mwh"])
W = D["act_wind"].astype(float)
p = D["ptf_usd"].astype(float)
Lm = L.rolling(24 * 30, min_periods=24 * 7).mean().shift(1)
pm = p.rolling(24 * 30, min_periods=24 * 7).mean().shift(1)
X = pd.DataFrame({"z": (L - solar_tot - W) / Lm, "solar_share": solar_tot / L, "wind_share": W / L,
                  "hour": D.index.hour, "month": D.index.month, "dow": D.index.dayofweek}, index=D.index)
y = p / pm
ok = X.notna().all(axis=1) & y.notna()
X, y, p, pm, solar_tot, W = X[ok], y[ok], p[ok], pm[ok], solar_tot[ok], W[ok]
tr = (X.index >= "2024-07-01") & (X.index < "2026-01-01"); te = X.index >= "2026-01-01"
feats = ["z", "hour", "month", "dow"]
params = dict(objective="l2", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, verbose=-1,
              monotone_constraints=[1, 0, 0, 0], seed=3)
m = lgb.train(params, lgb.Dataset(X.loc[tr, feats], y[tr]), 600)
r_hat = pd.Series(m.predict(X[feats]), index=X.index).clip(lower=0)
p_hat = r_hat * pm

def metrics(mask, label):
    pa, ph, s, w = p[mask], p_hat[mask], solar_tot[mask], W[mask]
    out = dict(period=label, mae_usd=float(np.abs(pa - ph).mean()),
               solar_capture_actual=float((pa * s).sum() / s.sum() / pa.mean()),
               solar_capture_model=float((ph * s).sum() / s.sum() / ph.mean()),
               wind_capture_actual=float((pa * w).sum() / w.sum() / pa.mean()),
               wind_capture_model=float((ph * w).sum() / w.sum() / ph.mean()),
               zero_hours_actual=int((pa <= 1).sum()), zero_hours_model=int((ph <= 1).sum()),
               hour_profile_actual=pa.groupby(pa.index.hour).mean().round(2).tolist(),
               hour_profile_model=ph.groupby(ph.index.hour).mean().round(2).tolist(),
               solar_share_of_load=float(s.sum() / L.reindex(pa.index).sum()))
    print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if "profile" not in k})
    return out

res = dict(k=k, train=metrics(tr, "eğitim 2024-07..2025-12"), test=metrics(te, "test 2026-01..2026-08"))
# r'nin z'ye tepkisi (diğerleri ortalama) — rapor için eğri
grid = np.linspace(0.3, 1.3, 41)
curve = {}
for h in [3, 12, 19]:
    Xg = pd.DataFrame({"z": grid, "hour": h, "month": 5, "dow": 2})
    curve[h] = m.predict(Xg).round(3).tolist()
res["curve"] = dict(z=grid.round(3).tolist(), by_hour=curve)
m.save_model("out/price_shape_model.txt")
json.dump(res, open("out/price_shape.json", "w"), indent=1)
