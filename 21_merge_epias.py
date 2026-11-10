"""EPİAŞ indirmelerini saatlik tabloya çevirir ve eski veriyle karşılaştırır."""
import pandas as pd, numpy as np
E = "data/epias"
def ts(d):  # yerel duvar saati (Europe/Istanbul), tz-naive
    return pd.to_datetime(d["date"], utc=True).dt.tz_convert("Europe/Istanbul").dt.tz_localize(None).values
p = pd.read_parquet(f"{E}/ptf.parquet"); p.index = ts(p)
s = pd.read_parquet(f"{E}/smf.parquet"); s.index = ts(s)
r = pd.read_parquet(f"{E}/direction.parquet"); r.index = ts(r)
g = pd.read_parquet(f"{E}/generation.parquet"); g.index = ts(g)
c = pd.read_parquet(f"{E}/consumption.parquet"); c.index = ts(c)
for _x in (p, s, r, g, c):
    print("çift:", _x.index.duplicated().sum(), end="  ")
p, s, r, g, c = [x[~x.index.duplicated(keep="first")] for x in (p, s, r, g, c)]
print()
df = pd.DataFrame({"ptf_tl": p.price, "ptf_usd_e": p.priceUsd, "ptf_eur": p.priceEur,
                   "smf_tl": s.systemMarginalPrice, "direction_id": r.smpDirectionId, "direction": r.systemDirection,
                   "cons_rt": c.consumption})
gc = [x for x in g.columns if x not in ("date", "hour")]
df = df.join(g[gc].add_prefix("gen_"))
df = df[~df.index.duplicated(keep="last")].sort_index().loc["2016-01-01":"2026-10-28 23:00"]
full = pd.date_range(df.index.min(), df.index.max(), freq="h")
print("saat:", len(df), "beklenen:", len(full), "eksik:", len(full.difference(df.index)))
df = df.reindex(full); df.index.name = "ts"
df.to_parquet("data/epias_hourly.parquet")
old = pd.read_parquet("data/tr_hourly.parquet")
o = pd.concat([old.ptf_usd, df.ptf_usd_e], axis=1).dropna()
print("eski vs EPİAŞ PTF USD: MAE", round((o.ptf_usd - o.ptf_usd_e).abs().mean(), 4), "max", round((o.ptf_usd - o.ptf_usd_e).abs().max(), 3), "n", len(o))
o = pd.concat([old.consumption_mwh, df.cons_rt], axis=1).dropna()
print("tüketim fark %:", round(((o.cons_rt / o.consumption_mwh) - 1).abs().mean() * 100, 3))
print("yön kodları:", df.groupby(["direction_id", "direction"]).size().to_dict())
print(df.isna().sum()[df.isna().sum() > 0].to_dict())
y = df.groupby(df.index.year)
gen = y[[c for c in df.columns if c.startswith("gen_")]].sum() / 1e6
print(gen.round(1)[["gen_total", "gen_naturalGas", "gen_dammedHydro", "gen_river", "gen_lignite", "gen_importCoal", "gen_wind", "gen_sun"]].to_string())
