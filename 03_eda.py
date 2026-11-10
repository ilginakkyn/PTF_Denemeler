import pandas as pd, numpy as np, json
df = pd.read_parquet("data/tr_hourly.parquet")
p = df.loc["2019":"2026-08-27","ptf_usd"]
res = {}
# zero / near-zero hours
z = (p<=1).groupby(p.index.year).sum(); res["zero_hours"]=z.to_dict()
print("Sıfır(<=1$) fiyatlı saat sayısı:", z.to_dict())
# monthly
mon = p.resample("MS").mean(); res["monthly"] = {k.strftime("%Y-%m"):round(v,2) for k,v in mon.items()}
print(mon.loc["2025-06":].round(1).to_string())
# hourly profile by year (normalized to yearly mean)
prof = p.groupby([p.index.year, p.index.hour]).mean().unstack(0)
prof_n = prof/prof.mean()
res["hour_profile_norm"] = {int(y): prof_n[y].round(3).tolist() for y in prof_n.columns}
print("\nSaatlik profil (yıllık ort.=1):"); print(prof_n.round(2).iloc[[3,7,10,12,14,17,19,21,23]].to_string())
# solar capture price: proxy using population-weighted surface solar radiation (2019-2025), and actual solar 2024-07+
s = df.loc["2019":"2025","ssr_popw_wm2"].clip(lower=0); pp = df.loc["2019":"2025","ptf_usd"]
cap = (s*pp).groupby(s.index.year).sum()/s.groupby(s.index.year).sum()
base = pp.groupby(pp.index.year).mean()
cr = (cap/base)
print("\nGüneş yakalama fiyatı (ışınım ağırlıklı) / baz:"); print(pd.DataFrame({"capture":cap.round(1),"base":base.round(1),"ratio":cr.round(3)}).to_string())
# actual solar generation weighted (licensed solar, 2024-07..2026-08)
a = df.loc["2024-07":"2026-08-27",["act_solar","ptf_usd","act_wind"]].dropna()
for k,g in a.groupby(a.index.year):
    print(k, "gerçek GES ağırlıklı capture ratio:", round((g.act_solar*g.ptf_usd).sum()/g.act_solar.sum()/g.ptf_usd.mean(),3),
          " rüzgar:", round((g.act_wind*g.ptf_usd).sum()/g.act_wind.sum()/g.ptf_usd.mean(),3))
res["capture_ratio_proxy"] = cr.round(4).to_dict()
# daily spread (max-min) as battery arbitrage potential
d = p.resample("D").agg(["max","min"]); sp = (d["max"]-d["min"]).groupby(d.index.year).mean()
print("\nOrtalama günlük max-min fark ($/MWh):", sp.round(1).to_dict()); res["daily_spread"]=sp.round(2).to_dict()
# best 4h vs worst 4h per day
def tb(x):
    v=np.sort(x.values); return v[-4:].mean()-v[:4].mean()
sp4 = p.resample("D").apply(tb).groupby(lambda t:t.year).mean(); print("En pahalı 4s - en ucuz 4s:", sp4.round(1).to_dict()); res["spread4"]=sp4.round(2).to_dict()
# solar ramp: actual solar capacity factor growth proxy: peak solar per month
ms = df.act_solar.resample("MS").max().dropna(); print("\nAylık max lisanslı GES üretimi (MWh/h):"); print(ms.round(0).to_string())
json.dump({str(k):v for k,v in res.items()}, open("out/eda.json","w"), default=str, indent=1)
