"""Kaynak repolardaki ham verileri tek, temiz saatlik veri setinde birleştirir.
Pickle dosyaları güvenlik nedeniyle okunmaz; yalnız CSV/Parquet/XLSX."""
import pandas as pd, numpy as np, json
EXT = "/home/claude/ext"
OUT = "/home/claude/proj/data"

# 1) PTF (USD/MWh) + yük tahmini + KGÜP, 2019-01-01 -> 2026-08-27  (EPİAŞ kaynaklı)
p = pd.read_csv(f"{EXT}/beratkrts_electricity-price-forecasting-turkish-day-ahead-market/experiments/notebooks/07_lago_protocol/tr_epf_ext.csv", index_col=0, parse_dates=True)
p.columns = ["ptf_usd", "load_plan_mw", "kgup_mw"]

# 2) Puant backtest: gerçekleşen PTF (TL), tüketim, rüzgar, güneş 2024-07 -> 2026-10
b = pd.read_parquet(f"{EXT}/emrtunali_puant/data/backtest.parquet")
b["time"] = b["time"].dt.tz_localize(None)
act = b.pivot_table(index="time", columns="target", values="actual", aggfunc="first")
act.columns = [f"act_{c}" for c in act.columns]
fc = b.pivot_table(index="time", columns="target", values="q50", aggfunc="first")
fc.columns = [f"puant_q50_{c}" for c in fc.columns]

# 3) Tüketim 2016-2025 + hava (nüfus ağırlıklı)
l = pd.read_csv(f"{EXT}/ahalikoguz_turkiye-electricity-demand-forecasting/data/processed/tr_load_hourly.csv", parse_dates=["datetime_local"])
l = l.set_index("datetime_local")[["consumption_mwh","is_imputed","is_holiday","holiday_name","is_half_day_holiday","t2m_popw_c","hdh18_popw","cdh22_popw","ssr_popw_wm2"]]

idx = pd.date_range("2016-01-01", act.index.max(), freq="h")
df = pd.DataFrame(index=idx).join(l).join(p).join(act).join(fc)
df.index.name = "ts"
df.to_parquet(f"{OUT}/tr_hourly.parquet")
print(df.shape)
print(df.notna().groupby(df.index.year).mean().round(2).T.to_string())
