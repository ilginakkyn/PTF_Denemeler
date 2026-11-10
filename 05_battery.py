"""Batarya arbitrajı: gerçek PTF (USD/MWh) üzerinde günlük LP.
Stratejiler aynı günü farklı fiyat bilgisiyle planlar, hepsi GERÇEK fiyatla uzlaştırılır:
  perfect   : mükemmel öngörü (üst sınır)
  model     : bizim LightGBM q50 tahmini (dürüst walk-forward)
  naive_D   : "yarın = bugün" (aynı saat, D günü)
  rule      : sabit saatler (şarj/deşarj saatleri test öncesi 12 aydan seçildi)
"""
import numpy as np, pandas as pd, json
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

bt = pd.read_parquet("out/ptf_backtest.parquet")
full = pd.read_parquet("data/tr_hourly.parquet")["ptf_usd"]

def plan_day(price, P=1.0, E=2.0, rte=0.88, cycles=1.0, deg=4.0):
    """24 saatlik LP. Değişkenler: c(24), d(24), s(24). SoC gün başı = gün sonu = 0."""
    n = len(price); ec = ed = np.sqrt(rte)
    nv = 3 * n
    cost = np.concatenate([price, -price + deg, np.zeros(n)])      # min: p*c - p*d + deg*d
    Aeq = lil_matrix((n + 1, nv)); beq = np.zeros(n + 1)
    for t in range(n):
        Aeq[t, 2 * n + t] = 1.0
        if t > 0:
            Aeq[t, 2 * n + t - 1] = -1.0
        Aeq[t, t] = -ec
        Aeq[t, n + t] = 1.0 / ed
    Aeq[n, 2 * n + n - 1] = 1.0                                    # gün sonu SoC = 0
    Aub = lil_matrix((1, nv)); Aub[0, n:2 * n] = 1.0; bub = [E * cycles]  # günlük deşarj sınırı
    bounds = [(0, P)] * (2 * n) + [(0, E)] * n
    r = linprog(cost, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    return r.x[:n], r.x[n:2 * n]

def simulate(signal_col, P, E, rte=0.88, cycles=1.0, deg=4.0, rule_hours=None):
    out = []
    for day, g in bt.groupby(bt.index.normalize()):
        if len(g) != 24:
            continue
        act = g["actual"].to_numpy()
        if rule_hours is not None:
            ch, dh = rule_hours
            c = np.zeros(24); d = np.zeros(24)
            hrs = int(round(E / P))
            c[ch[:hrs]] = P; d[dh[:hrs]] = P * rte          # şarj edilen enerjinin rte kadarı geri gelir
        else:
            sig = g[signal_col].to_numpy()
            if np.isnan(sig).any():
                sig = pd.Series(sig).ffill().bfill().to_numpy()
            sig = sig + 1e-4 * np.arange(24)          # deterministik eşitlik bozucu (çözücüden bağımsız sonuç)
            c, d = plan_day(sig, P, E, rte, cycles, deg)
        rev = float(np.sum(act * (d - c)) - deg * d.sum())
        out.append((day, rev, d.sum()))
    o = pd.DataFrame(out, columns=["day", "rev", "mwh_out"]).set_index("day")
    return o

# kural saatleri: test öncesi 12 ay (2023-09..2024-08) ortalama saatlik profilinden
pre = full.loc["2023-09-01":"2024-08-31"]
prof = pre.groupby(pre.index.hour).mean()
ch = list(prof.sort_values().index[:4]); ch.sort()
# deşarj: en pahalı ardışık olmayabilir; en pahalı saatler
dh = list(prof.sort_values(ascending=False).index[:4]); dh.sort()
rule_hours = (sorted(prof.sort_values().index[:4], key=lambda h: prof[h]), sorted(prof.sort_values(ascending=False).index[:4], key=lambda h: -prof[h]))

res = {}
daily = {}
for name, (P, E) in {"2h": (1.0, 2.0), "4h": (1.0, 4.0)}.items():
    for strat, col in [("perfect", "actual"), ("model", "ensemble"), ("model_q50", "lgb_q50"), ("naive_D", "naive_D"), ("rule", None)]:
        o = simulate(col, P, E, rule_hours=rule_hours if strat == "rule" else None)
        yrs = len(o) / 365.25
        res[f"{name}_{strat}"] = dict(usd_per_mw_year=round(o.rev.sum() / yrs, 0),
                                       cycles_per_year=round(o.mwh_out.sum() / E / yrs, 1), days=len(o))
        daily[f"{name}_{strat}"] = o.rev
        print(name, strat, res[f"{name}_{strat}"], flush=True)

for name in ["2h", "4h"]:
    pf = res[f"{name}_perfect"]["usd_per_mw_year"]
    for s in ["model", "model_q50", "naive_D", "rule"]:
        res[f"{name}_{s}"]["share_of_perfect"] = round(res[f"{name}_{s}"]["usd_per_mw_year"] / pf, 3)
res["rule_hours"] = {"charge": [int(h) for h in rule_hours[0]], "discharge": [int(h) for h in rule_hours[1]]}

# yıllara göre (test dönemi 12'şer ay): 2024-09..2025-08 ve 2025-09..2026-08
dd = pd.DataFrame(daily)
per = dd.groupby(np.where(dd.index < "2025-09-01", "Eyl24-Ağu25", "Eyl25-Ağu26")).sum().round(0)
res["by_period"] = per.to_dict()
print(per.T)
dd.to_csv("out/battery_daily.csv")
json.dump(res, open("out/battery.json", "w"), indent=1, default=str)
