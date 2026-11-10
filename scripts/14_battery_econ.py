"""Batarya yatırım ekonomisi (USD). Gelir = backtest'te ölçülen arbitraj (model tahminiyle, son 12 ay).
Capex: BNEF Ara-2025 anahtar teslim (Avrupa ort. $177/kWh; 2s $124, 4s $110 küresel) -> Türkiye için varsayım."""
import json, numpy as np
bat = json.load(open("out/battery.json"))
yrs = 15; wacc = 0.10; deg = 0.02
res = {}
for dur, capex_kwh, rev_key in [("2h", 180, "2h_model"), ("4h", 150, "4h_model")]:
    E = 2 if dur == "2h" else 4
    capex = capex_kwh * 1000 * E * 1.10          # +%10 şebeke bağlantısı/arazi
    om = 0.02 * capex
    import pandas as pd
    dd = pd.read_csv("out/battery_daily.csv", index_col=0, parse_dates=True)[rev_key]
    last = dd[dd.index >= "2025-09-01"]
    rev_last12 = float(last.sum() * 365 / len(last))            # 361 gün -> 365 gün
    rev_avg = bat[rev_key]["usd_per_mw_year"]
    def npv(rev, extra=0.0):
        cf = [-capex] + [(rev * (1 - deg) ** t + extra - om) / (1 + wacc) ** (t + 1) for t in range(yrs)]
        return sum(cf)
    # başabaş için gereken ek (kapasite/yan hizmet) geliri
    lo, hi = 0, 200000
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if npv(rev_last12, mid) < 0 else (lo, mid)
    # sadece arbitrajla başabaş capex ($/kWh)
    ann = sum((rev_last12 * (1 - deg) ** t - 0) / (1 + wacc) ** (t + 1) for t in range(yrs))
    be_capex_kwh = ann / (1.10 * 1000 * E * (1 + 0.02 * sum(1 / (1 + wacc) ** (t + 1) for t in range(yrs))))
    res[dur] = dict(capex_usd_per_mw=round(capex), om_usd_per_mw_yr=round(om), arbitrage_rev_last12=rev_last12,
                    arbitrage_rev_2y_avg=rev_avg, simple_payback_yrs=round(capex / (rev_last12 - om), 1),
                    cum_cash_yr15_undiscounted=round(sum(rev_last12 * (1 - deg) ** t - om for t in range(yrs)) - capex),
                    npv_arbitrage_only=round(npv(rev_last12)), extra_revenue_needed_usd_per_mw_yr=round(hi),
                    breakeven_capex_usd_per_kwh_arbitrage_only=round(be_capex_kwh, 1))
    print(dur, res[dur])
res["assumptions"] = dict(life_yrs=yrs, wacc_usd=wacc, degradation=deg, grid_adder=0.10, om_share=0.02,
                          capex_2h_usd_kwh=180, capex_4h_usd_kwh=150, rte=0.88,
                          source="BNEF Battery Storage System Cost Survey, Dec 2025 (energy-storage.news): global ort. $117/kWh, Avrupa $177/kWh")
json.dump(res, open("out/battery_econ.json", "w"), indent=1)
