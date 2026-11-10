"""Gün öncesi PTF tahmini — dürüst (walk-forward) backtest.

Bilgi seti: D+1 günü için teklif D günü 12:30'dan önce verilir. O anda bilinenler:
  * D gününün tüm saatlik PTF'leri (D-1 günü 14:00'te yayımlanır)  -> fiyat gecikmeleri >= 24 saat
  * TEİAŞ yük tahmin planı (D+1 için)                              -> load_plan_mw(t)
  * KGÜP yalnız D günü ve öncesi (D+1 KGÜP'ü piyasa sonrası oluşur; kullanmak sızıntı olur)
Hava verisi kullanılmadı: elimizdeki gerçekleşen hava, o gün bilinemeyeceği için sızıntı olurdu.

Fiyatlar USD/MWh: TL enflasyonu modelin "öğrendiği" şey olmasın diye.
Test: 2024-09-01 .. 2026-08-27, her ayın başında yeniden eğitim, yalnız geçmiş veriyle.
"""
import warnings, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

df = pd.read_parquet("data/tr_hourly.parquet").loc["2019-01-01":"2026-08-27 23:00"]
y = df["ptf_usd"].astype(float)

# --- tatil bayrakları: 2019-2025 veri setinden, 2026 elle (resmi tatiller)
hol = df["is_holiday"].fillna(0).astype(float)
half = df["is_half_day_holiday"].fillna(0).astype(float)
h26 = ["2026-01-01", "2026-03-20", "2026-03-21", "2026-03-22", "2026-04-23", "2026-05-01",
       "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30", "2026-07-15", "2026-08-30"]
for d in h26:
    hol[hol.index.normalize() == pd.Timestamp(d)] = 1.0   # aralık dışı tarih eklemez
for d in ["2026-03-19", "2026-05-26"]:
    half.loc[f"{d} 13:00":f"{d} 23:00"] = 1.0

X = pd.DataFrame(index=df.index)
for L in [24, 25, 47, 48, 72, 96, 120, 144, 168, 336]:
    X[f"lag{L}"] = y.shift(L)
day = y.resample("D")
dmean, dmin, dmax = day.mean(), day.min(), day.max()
dstd = day.std()
def daily_feat(s, lag_days):
    s2 = s.copy(); s2.index = s2.index + pd.Timedelta(days=lag_days)
    return s2.reindex(df.index.normalize()).to_numpy()
X["dmean_D"] = daily_feat(dmean, 1)      # hedef günden bir önceki gün (D) ortalaması
X["dmin_D"] = daily_feat(dmin, 1)
X["dmax_D"] = daily_feat(dmax, 1)
X["dstd_D"] = daily_feat(dstd, 1)
X["dmean_D-1"] = daily_feat(dmean, 2)
wk = y.rolling(168).mean().shift(24)
X["wmean"] = wk
X["m28"] = y.rolling(24 * 28).mean().shift(24)
# aynı saatte son 7 günün ortalaması / medyanı
X["samehour7"] = sum(y.shift(24 * k) for k in range(1, 8)) / 7
X["last_hour_D"] = daily_feat(y[y.index.hour == 23].resample("D").last(), 1)

lp = df["load_plan_mw"].astype(float)
X["load_plan"] = lp
X["load_plan_d_mean"] = lp.groupby(lp.index.normalize()).transform("mean")
X["load_plan_vs_lag168"] = lp - lp.shift(168)
X["load_plan_vs_lag24"] = lp - lp.shift(24)
X["load_plan_rel"] = lp / lp.rolling(24 * 28).mean().shift(24)
kg = df["kgup_mw"].astype(float)
X["kgup_lag24"] = kg.shift(24)
X["kgup_lag168"] = kg.shift(168)
X["kgup_minus_load_lag24"] = (kg - lp).shift(24)

X["hour"] = df.index.hour
X["dow"] = df.index.dayofweek
X["month"] = df.index.month
doy = df.index.dayofyear
X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
X["holiday"] = hol.values
X["half_holiday"] = half.values
X["holiday_lag24"] = hol.shift(24).values

TEST_START, TEST_END = "2024-09-01", "2026-08-27 23:00"
months = pd.date_range(TEST_START, "2026-08-01", freq="MS")
feats = list(X.columns)

PARAMS = dict(objective="l1", learning_rate=0.03, num_leaves=63, min_data_in_leaf=60,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)

def fit_predict(train_mask, test_mask, objective="l1", alpha=None, scale_target=True, rounds=900):
    Xtr, ytr = X[train_mask], y[train_mask]
    ok = Xtr.notna().all(axis=1) & ytr.notna()
    Xtr, ytr = Xtr[ok], ytr[ok]
    level_tr = Xtr["wmean"].clip(lower=5)
    level_te = X.loc[test_mask, "wmean"].clip(lower=5)
    target = ytr / level_tr if scale_target else ytr
    p = dict(PARAMS, objective=objective)
    if alpha is not None:
        p["alpha"] = alpha
    # yakın geçmişe daha fazla ağırlık (yarı ömür ~ 1 yıl)
    age_days = (Xtr.index.max() - Xtr.index).days.values
    w = 0.5 ** (age_days / 365.0)
    m = lgb.train(p, lgb.Dataset(Xtr[feats], target, weight=w), num_boost_round=rounds)
    pred = m.predict(X.loc[test_mask, feats])
    return (pred * level_te.values if scale_target else pred), m

results = {k: [] for k in ["lgb_q50", "lgb_q10", "lgb_q90", "lgb_unscaled", "ridge"]}
imp = []
t0 = time.time()
for m0 in months:
    m1 = min(m0 + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23), pd.Timestamp(TEST_END))
    test = (X.index >= m0) & (X.index <= m1)
    train = X.index < m0 - pd.Timedelta(days=1)   # D+1 tahmini, D'nin sonuna kadar veri yok sayılır
    p50, model = fit_predict(train, test, "quantile", 0.5)
    p10, _ = fit_predict(train, test, "quantile", 0.1, rounds=600)
    p90, _ = fit_predict(train, test, "quantile", 0.9, rounds=600)
    pu, _ = fit_predict(train, test, "l1", scale_target=False)
    idx = X.index[test]
    results["lgb_q50"].append(pd.Series(p50, idx))
    results["lgb_q10"].append(pd.Series(p10, idx))
    results["lgb_q90"].append(pd.Series(p90, idx))
    results["lgb_unscaled"].append(pd.Series(pu, idx))
    imp.append(pd.Series(model.feature_importance("gain"), index=feats))
    # LEAR benzeri doğrusal model: son 2 yıl, saat-bazlı ayrı Ridge
    rp = pd.Series(np.nan, idx)
    for h in range(24):
        trh = train & (X.index >= m0 - pd.Timedelta(days=730)) & (X.index.hour == h)
        teh = test & (X.index.hour == h)
        cols = [c for c in feats if c not in ("hour",)]
        A = X.loc[trh, cols]; b = y[trh]; ok = A.notna().all(axis=1) & b.notna()
        sc = StandardScaler().fit(A[ok])
        r = Ridge(alpha=3.0).fit(sc.transform(A[ok]), b[ok])
        rp.loc[X.index[teh]] = r.predict(sc.transform(X.loc[teh, cols].ffill().fillna(A[ok].mean())))
    results["ridge"].append(rp)
    print(f"{m0:%Y-%m} tamam  ({time.time()-t0:.0f}s)", flush=True)

pred = pd.DataFrame({k: pd.concat(v) for k, v in results.items()})
pred["actual"] = y.loc[pred.index]
pred["naive_D"] = X.loc[pred.index, "lag24"]            # aynı saat, D günü
pred["naive_week"] = X.loc[pred.index, "lag168"]        # aynı saat, geçen hafta
pred["naive_mix"] = X.loc[pred.index, "samehour7"]
pred["ensemble"] = pred[["lgb_q50", "lgb_unscaled", "ridge"]].mean(axis=1)
# Puant (bağımsız, kamuya açık model) aynı saatler için: TL q50 -> USD (o saatin TL/USD oranı ile)
fx = (df["act_price"] / df["ptf_usd"]).where(df["ptf_usd"] > 1)
fx = fx.groupby(fx.index.normalize()).transform("median").ffill()
pred["puant_ext"] = (df["puant_q50_price"] / fx).loc[pred.index]
pred["fx_usdtry"] = fx.loc[pred.index]
pred.to_parquet("out/ptf_backtest.parquet")
imp_mean = pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False)
imp_mean.to_csv("out/feature_importance.csv")
print("bitti", time.time() - t0)
