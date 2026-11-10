"""Toplayıcı portföy etkisi: dağınık GES'lerin tek portföyde toplanmasıyla dengesizliğin azalması.

Gerçek veri: Türkiye lisanslı GES saatlik gerçekleşen üretimi ve bağımsız bir gün-öncesi tahmini
(Puant q50), 2024-07..2026-10. Ulusal ölçekteki tahmin hatası GERÇEK; bölgesel ve tesis-yerel
sapmalar bu ortak sinyalin üzerine stokastik olarak eklenir (parametreler varsayımdır, duyarlılık
analizi yapılır). Fiyatlar gerçek PTF (USD/MWh).

Model, tesis i, saat t (kapasite 1 MW):
  gerçek_i,t  = CF_ulusal,t * (1 + e_bölge(i),t + e_yerel_i,t)    (>= 0, <= 1)
  tahmin_i,t  = CF_tahmin_ulusal,t                                  (küçük üretici ulusal profili kullanır)
  e_*         : saatlik AR(1), ρ = 0.9 (bulut sistemleri saatler boyu sürer)
"""
import numpy as np, pandas as pd, json
rng = np.random.default_rng(42)
df = pd.read_parquet("data/tr_hourly.parquet").loc["2024-07-01":"2026-08-27 23:00"]
A = df["act_solar"].astype(float); F = df["puant_q50_solar"].astype(float)
ok = A.notna() & F.notna(); A, F = A[ok], F[ok]
# kapasite vekili: 30 günlük kayan maksimumun 0.85'e bölünmesi (öğle saatinde CF ~0.85)
cap = A.rolling(24 * 30, min_periods=24 * 7).max().bfill() / 0.85
cfA = (A / cap).clip(0, 1); cfF = (F / cap).clip(0, 1)
ptf = df.loc[cfA.index, "ptf_usd"].ffill().to_numpy()
daylight = cfA.to_numpy() + cfF.to_numpy() > 0.01
U = pd.read_parquet("data/imbalance_unit_costs.parquet").reindex(cfA.index)
up_c = U["up_cost_usd"].ffill().to_numpy(); dn_c = U["down_cost_usd"].ffill().to_numpy()
is26 = cfA.index >= "2026-01-01"
T = len(cfA)

nat_err = (cfA - cfF).to_numpy()
nat_nmae = np.abs(nat_err[daylight]).mean()
print(f"Ulusal (gerçek) gün-öncesi hata, gündüz nMAE = {nat_nmae:.3%} (kapasiteye oranla)")

def ar1(n, k, rho, sd):
    e = np.zeros((n, k)); z = rng.standard_normal((n, k)) * sd * np.sqrt(1 - rho ** 2)
    e[0] = rng.standard_normal(k) * sd
    for t in range(1, n):
        e[t] = rho * e[t - 1] + z[t]
    return e

def run(n_sites=40, n_regions=7, sd_reg=0.18, sd_loc=0.12, rho=0.9):
    reg = ar1(T, n_regions, rho, sd_reg)
    loc = ar1(T, n_sites, rho, sd_loc)
    region_of = np.arange(n_sites) % n_regions
    act = np.clip(cfA.to_numpy()[:, None] * (1 + reg[:, region_of] + loc), 0, 1)
    fc = np.repeat(cfF.to_numpy()[:, None], n_sites, axis=1)
    err = act - fc                                  # MW (1 MW tesis)
    site_nmae = np.abs(err[daylight]).mean()
    standalone = np.abs(err).sum(axis=1)            # her tesis kendi dengesizliğini öder
    pooled = np.abs(err.sum(axis=1))                # toplayıcı tek dengeden sorumlu
    energy = act.sum()
    out = dict(n_sites=n_sites, site_daylight_nMAE=round(site_nmae, 4),
               imbalance_share_standalone=round(standalone.sum() / energy, 4),
               imbalance_share_pooled=round(pooled.sum() / energy, 4),
               reduction=round(1 - pooled.sum() / standalone.sum(), 4))
    # parasal: dengesizlik MWh başına maliyet = c * PTF ; c senaryoları
    for c in (0.06, 0.12, 0.20):
        out[f"cost_usd_per_mwh_gen_standalone_c{int(c*100)}"] = round((standalone * ptf * c).sum() / energy, 3)
        out[f"cost_usd_per_mwh_gen_pooled_c{int(c*100)}"] = round((pooled * ptf * c).sum() / energy, 3)
    # GERÇEK birim maliyetlerle (SMF/PTF, yön; 2026 kuralı dahil)
    def real(e):
        return np.where(e > 0, e * up_c[:, None] if e.ndim == 2 else e * up_c, -e * (dn_c[:, None] if e.ndim == 2 else dn_c))
    st = real(err).sum(axis=1); po = real(err.sum(axis=1))
    out["real_cost_standalone"] = round(st.sum() / energy, 3); out["real_cost_pooled"] = round(po.sum() / energy, 3)
    out["real_saving"] = round((st.sum() - po.sum()) / energy, 3)
    gen = act.sum(axis=1)
    out["real_cost_standalone_2026"] = round(st[is26].sum() / gen[is26].sum(), 3); out["real_cost_pooled_2026"] = round(po[is26].sum() / gen[is26].sum(), 3)
    rev = (act * ptf[:, None]).sum()
    out["real_saving_pct_revenue"] = round((st.sum() - po.sum()) / rev, 4)
    return out

results = {"national_real_daylight_nMAE": round(float(nat_nmae), 4), "hours": int(T), "runs": []}
# 1) Portföy büyüklüğü
for n in [1, 5, 10, 20, 40, 80, 160]:
    r = run(n_sites=n); r["case"] = "size"; results["runs"].append(r); print(r)
# 2) Duyarlılık: yerel/bölgesel sapma şiddeti (tek tesis hatası %7..%14 aralığı)
for sr, sl, lab in [(0.10, 0.07, "düşük"), (0.18, 0.12, "baz"), (0.26, 0.18, "yüksek")]:
    r = run(n_sites=40, sd_reg=sr, sd_loc=sl); r["case"] = f"sens_{lab}"; results["runs"].append(r); print(lab, r)
# 3) Bölge çeşitliliği: hepsi tek bölgede vs 7 bölge
r = run(n_sites=40, n_regions=1); r["case"] = "one_region"; results["runs"].append(r); print("tek bölge", r)
json.dump(results, open("out/aggregator.json", "w"), indent=1)
