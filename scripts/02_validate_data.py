import pandas as pd, numpy as np, json
df = pd.read_parquet("data/tr_hourly.parquet")
print("== 1) Tüketim: iki bağımsız kaynak karşılaştırması (2024-07..2025-12)")
o = df[["consumption_mwh","act_consumption"]].dropna()
d = (o.consumption_mwh - o.act_consumption)
print(f"n={len(o)}  ort fark={d.mean():.1f} MWh  MAE={d.abs().mean():.1f}  korelasyon={o.corr().iloc[0,1]:.5f}  |fark|>1% oranı={(d.abs()/o.act_consumption>0.01).mean():.3%}")

print("\n== 2) Yıllık toplam tüketim (TWh), EPİAŞ gerçek zamanlı tüketim")
print((df.consumption_mwh.groupby(df.index.year).sum()/1e6).round(1).to_string())

print("\n== 3) PTF TL / PTF USD -> ima edilen USDTRY")
o = df[["act_price","ptf_usd"]].dropna(); o = o[o.ptf_usd>1]
r = (o.act_price/o.ptf_usd)
day = r.groupby(r.index.date).agg(["median","std"])
print("günlük kur std'si (gün içi tutarlılık) medyan:", round(day["std"].median(),4))
m = r.groupby(r.index.to_period("M")).median().round(2)
print(m.to_string())

print("\n== 4) Bağımsız ayna (EPİAŞ günlük ağırlıklı PTF, TL/kWh) vs saatlik TL seri günlük ort")
j = json.load(open("/home/claude/ext/06maslan27_fatura-dedektifi-veri/data/piyasa/ptf-yekdem.json"))
s = [x for x in j["priceSeries"] if x["id"]=="epias-ptf-gunluk"][0]["points"]
mir = pd.Series({pd.Timestamp(x["date"]): float(x["price"])*1000 for x in s})
ours = df.act_price.resample("D").mean()
c = pd.concat([mir.rename("ayna_agirlikli"), ours.rename("bizim_basit_ort")], axis=1).dropna()
c["fark_%"] = (c.bizim_basit_ort/c.ayna_agirlikli-1)*100
print(c.describe().round(2).to_string()); print(c.tail(5).round(1).to_string())

print("\n== 5) Yıllık ortalama PTF (USD/MWh, basit)")
print(df.ptf_usd.groupby(df.index.year).agg(["mean","min","max","count"]).round(1).to_string())
print("\n== 6) Saat boşlukları / NaN")
print(df.loc["2019":"2026-08-27", "ptf_usd"].isna().sum(), "NaN ptf_usd;", (df.loc["2019":"2026-08-27","ptf_usd"]<=0).sum(), "<=0 saat")
