"""Rapor (pano) için veri dosyalarını üretir: dash/*.csv"""
import json, os, numpy as np, pandas as pd
os.makedirs("dash", exist_ok=True)
D = pd.read_parquet("data/tr_hourly.parquet")
EP = pd.read_parquet("data/epias_hourly.parquet")
p = EP.loc["2016":"2026-10-28", "ptf_usd_e"].astype(float)       # EPİAŞ, doğrudan
END_LABEL = "Oca–28 Eki"

# 1) yıllık özet
official = {2019: 45.8, 2020: 39.6, 2021: 57.2, 2022: 151.0, 2023: 92.0, 2024: 68.0, 2025: 66.2}
official_tl = {2019: 260.32, 2020: 278.72, 2021: 508.10, 2022: 2510.72, 2023: 2189.30, 2024: 2235.52, 2025: 2617.83}
rows = []
for y in range(2016, 2027):
    py = p[p.index.year == y]
    tl = EP.loc[str(y), "ptf_tl"].astype(float)
    d = py.resample("D").apply(lambda x: np.sort(x.values)[-4:].mean() - np.sort(x.values)[:4].mean())
    s = D.loc[str(y), "ssr_popw_wm2"].clip(lower=0) if y <= 2025 else None
    cap = float((s * py.reindex(s.index)).sum() / s.sum() / py.mean()) if s is not None else None
    a = pd.concat([EP.loc[str(y), "gen_sun"].astype(float), py], axis=1).dropna()
    cap_act = float((a.iloc[:, 0] * a.iloc[:, 1]).sum() / a.iloc[:, 0].sum() / a.iloc[:, 1].mean()) if a.iloc[:, 0].sum() > 1e5 else None
    rows.append(dict(year=str(y), period=(END_LABEL if y == 2026 else "tam yıl"),
                     ptf_usd=round(py.mean(), 2), official_usd=official.get(y),
                     ptf_tl=round(tl.mean(), 1), official_tl=official_tl.get(y),
                     zero_hours=int((py <= 1).sum()), spread4=round(d.mean(), 2),
                     midday_ratio=round(py[py.index.hour == 12].mean() / py.mean(), 3),
                     evening_ratio=round(py[py.index.hour.isin([19, 20, 21])].mean() / py.mean(), 3),
                     solar_capture_irradiance=None if cap is None else round(cap, 3),
                     solar_capture_licensed=None if cap_act is None else round(cap_act, 3),
                     consumption_twh=round(EP.loc[str(y), "cons_rt"].astype(float).sum() / 1e6, 1)))
pd.DataFrame(rows).to_csv("dash/annual.csv", index=False)

# 2) saatlik profil (yıllık ortalamaya oranla)
hp = p.groupby([p.index.year, p.index.hour]).mean().unstack(0); hp = hp / hp.mean()
hp.index.name = "hour"
hp.reset_index().melt(id_vars="hour", var_name="year", value_name="ratio").assign(ratio=lambda x: x.ratio.round(3), year=lambda x: x.year.astype(str)).to_csv("dash/hour_profile.csv", index=False)

# 3) aylık PTF + hidro + dengesizlik
U = pd.read_parquet("data/imbalance_unit_costs.parquet")
m = pd.DataFrame({"ptf_usd": p.resample("MS").mean(),
                  "smf_usd": U.smf_usd.loc["2016":].resample("MS").mean(),
                  "hydro_twh": (EP.gen_dammedHydro + EP.gen_river).astype(float).loc["2016":].resample("MS").sum() / 1e6,
                  "deficit_share": (U.direction_id == 1).loc["2016":].resample("MS").mean()}).round(3)
m.index = m.index.strftime("%Y-%m"); m.index.name = "month"; m.reset_index().to_csv("dash/monthly.csv", index=False)

# 3b) yıllık üretim karışımı (EPİAŞ gerçek zamanlı, lisanslı) ve dengesizlik
g = EP.loc["2016":].select_dtypes("number"); gy = g.groupby(g.index.year)
mix = pd.DataFrame({"hidro": gy.gen_dammedHydro.sum() + gy.gen_river.sum(), "dogalgaz": gy.gen_naturalGas.sum() + gy.gen_lng.sum(),
                    "ithal_komur": gy.gen_importCoal.sum(), "yerli_komur": gy.gen_lignite.sum() + gy.gen_asphaltiteCoal.sum() + gy.gen_blackCoal.sum(),
                    "ruzgar": gy.gen_wind.sum(), "gunes_lisansli": gy.gen_sun.sum(),
                    "diger": gy.gen_geothermal.sum() + gy.gen_biomass.sum() + gy.gen_fueloil.sum() + gy.gen_wasteheat.sum() + gy.gen_naphta.sum()}) / 1e6
mix = mix.round(2); mix.index = mix.index.astype(str); mix.index.name = "year"
mix.reset_index().melt(id_vars="year", var_name="source", value_name="twh").to_csv("dash/gen_mix.csv", index=False)
imb = pd.read_csv("out/imbalance_annual.csv", dtype={"year": str})
imb.to_csv("dash/imbalance_annual.csv", index=False)

# 4) model metrikleri
ev = json.load(open("out/ptf_eval.json"))
names = {"naive_D": "Naif: bugünün aynı saati", "naive_week": "Naif: geçen hafta aynı saat", "naive_mix": "Naif: son 7 gün ort.",
         "ridge": "Doğrusal (saat bazlı Ridge)", "lgb_unscaled": "LightGBM", "lgb_q50": "LightGBM (göreli, medyan)",
         "ensemble": "Topluluk (bizim model)", "puant_ext": "Puant (bağımsız açık model)"}
mm = []
for k, v in ev["overall"].items():
    mm.append(dict(model=names[k], key=k, mae=round(v["MAE"], 2), rmse=round(v["RMSE"], 2), smape=round(v["sMAPE"], 1),
                   rmae=round(v["rMAE"], 3), mae_tl=ev["MAE_TL"][k],
                   mae_p1=round(ev["by_period_MAE"]["Eyl24-Ağu25"][k], 2), mae_p2=round(ev["by_period_MAE"]["Eyl25-Ağu26"][k], 2)))
_b1 = pd.read_parquet("out/ptf_backtest.parquet"); _b2 = pd.read_parquet("out/ptf_backtest_v2.parquet")
_c = _b1[["actual", "naive_D"]].join(_b2[["ensemble"]].rename(columns={"ensemble": "v2"})).dropna()
_c = _c.loc[_b1.dropna(subset=["naive_D", "ridge", "lgb_unscaled", "lgb_q50", "ensemble", "puant_ext", "naive_week", "naive_mix"]).index.intersection(_c.index)]
_e = _c.v2 - _c.actual; _fx = _b1.loc[_c.index, "fx_usdtry"]; _nmae = (_c.naive_D - _c.actual).abs().mean()
mm.append(dict(model="Topluluk + hidro/DGP girdileri (v2)", key="ensemble_v2", mae=round(_e.abs().mean(), 2), rmse=round(np.sqrt((_e**2).mean()), 2),
               smape=round((2 * _e.abs() / (_c.actual.abs() + _c.v2.abs() + 1e-9)).mean() * 100, 1), rmae=round(_e.abs().mean() / _nmae, 3),
               mae_tl=round(float((_e.abs() * _fx).mean()), 1), mae_p1=round(_e[_c.index < "2025-09-01"].abs().mean(), 2), mae_p2=round(_e[_c.index >= "2025-09-01"].abs().mean(), 2)))
pd.DataFrame(mm).to_csv("dash/model_metrics.csv", index=False)
pd.DataFrame({"hour": range(24), "ensemble": ev["mae_by_hour_ens"], "naive": ev["mae_by_hour_naive"]}).to_csv("dash/mae_hour.csv", index=False)
pd.DataFrame([dict(metric="q10_q90_raw", value=ev["q10_q90_coverage"]), dict(metric="q10_q90_calibrated", value=ev["q10_q90_coverage_calibrated"]),
              dict(metric="band_width_raw", value=ev["band_width_raw"]), dict(metric="band_width_cal", value=ev["band_width_cal"]),
              dict(metric="test_hours", value=ev["all_hours_n"])]).to_csv("dash/model_band.csv", index=False)
fi = pd.read_csv("out/feature_importance.csv", index_col=0).iloc[:, 0]; fi = (fi / fi.sum()).head(12).round(4)
fnames = {"wmean": "Son 7 gün ort. fiyat", "lag24": "Bugün aynı saat", "lag336": "2 hafta önce aynı saat", "load_plan_rel": "Yük tahmini / 28g ort.",
          "last_hour_D": "Bugünün son saati", "load_plan_vs_lag24": "Yük tahmini − dün", "hour": "Saat", "lag168": "Geçen hafta aynı saat",
          "samehour7": "Son 7 gün aynı saat ort.", "m28": "Son 28 gün ort.", "kgup_minus_load_lag24": "KGÜP − yük (bugün)", "load_plan_vs_lag168": "Yük tahmini − geçen hafta",
          "dmean_D": "Bugünün ortalaması", "dmax_D": "Bugünün maksimumu", "dmin_D": "Bugünün minimumu", "load_plan": "Yük tahmini", "doy_cos": "Mevsim (cos)", "doy_sin": "Mevsim (sin)",
          "dstd_D": "Bugünün oynaklığı", "kgup_lag24": "KGÜP (bugün)", "lag25": "Bugün bir önceki saat", "lag48": "Dün aynı saat", "load_plan_d_mean": "Yarının ort. yük tahmini", "dow": "Haftanın günü", "month": "Ay"}
pd.DataFrame({"feature": [fnames.get(k, k) for k in fi.index], "share": fi.values}).to_csv("dash/feature_importance.csv", index=False)
bt = pd.read_parquet("out/ptf_backtest.parquet").loc["2026-08-14":"2026-08-27"]
pd.DataFrame({"ts": bt.index.strftime("%Y-%m-%dT%H:00"), "actual": bt.actual.round(2), "forecast": bt.ensemble.round(2),
              "q10": bt.q10_cal.round(2), "q90": bt.q90_cal.round(2), "naive": bt.naive_D.round(2)}).to_csv("dash/forecast_sample.csv", index=False)

# 5) toplayıcı
ag = json.load(open("out/aggregator.json"))
ar = pd.DataFrame(ag["runs"])
labels = {"size": "Portföy", "sens_düşük": "Düşük yerel hata", "sens_baz": "Baz", "sens_yüksek": "Yüksek yerel hata", "one_region": "40 tesis tek bölgede"}
ar["case_label"] = ar.case.map(labels)
ar["key"] = ar.case + "_" + ar.n_sites.astype(str)
keep = ["key", "case", "case_label", "n_sites", "site_daylight_nMAE", "imbalance_share_standalone", "imbalance_share_pooled", "reduction"] + [c for c in ar.columns if c.startswith("cost_") or c.startswith("real_")]
ar[keep].to_csv("dash/aggregator.csv", index=False)
pd.DataFrame([dict(metric="national_daylight_nMAE", value=ag["national_real_daylight_nMAE"]), dict(metric="hours", value=ag["hours"])]).to_csv("dash/aggregator_meta.csv", index=False)

# 6) batarya
b = json.load(open("out/battery.json"))
snames = {"perfect": "Mükemmel öngörü (üst sınır)", "model": "Bizim tahmin", "model_q50": "Bizim tahmin (medyan)", "naive_D": "Naif: yarın = bugün", "rule": "Sabit saat kuralı"}
br = []
for dur in ["2h", "4h"]:
    for s in ["perfect", "model", "naive_D", "rule"]:
        k = f"{dur}_{s}"
        br.append(dict(key=k, duration=dur, strategy=snames[s], usd_per_mw_year=b[k]["usd_per_mw_year"],
                       share_of_perfect=b[k].get("share_of_perfect", 1.0), p1=b["by_period"][k]["Eyl24-Ağu25"], p2=b["by_period"][k]["Eyl25-Ağu26"]))
pd.DataFrame(br).to_csv("dash/battery.csv", index=False)
be = json.load(open("out/battery_econ.json"))
pd.DataFrame([dict(duration=d, **{k: (round(v) if isinstance(v, float) and abs(v) > 100 else v) for k, v in be[d].items()}) for d in ["2h", "4h"]]).to_csv("dash/battery_econ.csv", index=False)

# 7) 2035 senaryolar
sc = json.load(open("out/scenarios_2035.json"))
dn = {"dusuk": "Düşük (%2.5/yıl)", "baz": "Baz — UEP (510 TWh)", "yuksek": "Yüksek (%4.7/yıl)"}
sn = {"uep2022": "UEP 2022 (82.5 GW)", "mevcut_hiz": "Mevcut hız (95 GW)", "hedef_120": "120 GW hedefi"}
tn = {"yok": "Depolama yok", "uep_7.5GW_2s": "7.5 GW / 2 saat (UEP)", "yuksek_20GW_4s": "20 GW / 4 saat"}
srows, hrows = [], []
for r in sc:
    lab = r["label"]
    if lab == "2025_model":
        dem, sup, sto, extra = "—", "—", "—", "2025 kalibrasyon"
    else:
        parts = lab.split("|"); dem, sup, sto = parts[:3]; extra = parts[3] if len(parts) > 3 else "ortalama hidroloji"
        extra = {"kurak": "kurak yıl (50 TWh hidro)", "yagisli": "yağışlı yıl (85 TWh hidro)", "co2=30": "karbon $30/t", "co2=60": "karbon $60/t"}.get(extra, extra)
    srows.append(dict(key=lab, demand=dn.get(dem, dem), supply=sn.get(sup, sup), storage=tn.get(sto, sto), variant=extra,
                      demand_twh=round(r["demand_twh"], 1), avg_price=round(r["avg_price"], 1), solar_capture_ratio=round(r["solar_capture_ratio"], 3),
                      wind_capture_ratio=round(r["wind_capture_ratio"], 3), zero_price_hours=r["zero_price_hours"], curtail_twh=round(r["curtail_twh"], 1),
                      curtail_share_vre=round(r["curtail_share_vre"], 3), vre_share=round(r["vre_share_of_demand"], 3), gas_twh=round(r["gas_twh"], 1),
                      coal_twh=round(r["coal_twh"], 1), spread4=round(r["spread4_avg"], 1),
                      unserved_gwh=round(r["unserved_mwh"] / 1e3, 1)))
    for h, v in enumerate(r["hour_profile"]):
        hrows.append(dict(key=lab, hour=h, price=v))
pd.DataFrame(srows).to_csv("dash/scenarios.csv", index=False)
a25 = D.loc["2025", "ptf_usd"]
for h, v in a25.groupby(a25.index.hour).mean().round(2).items():
    hrows.append(dict(key="2025_actual", hour=h, price=v))
pd.DataFrame(hrows).to_csv("dash/scenario_hours.csv", index=False)

# 8) fiyat-şekil modeli doğrulaması (2026 dış örneklem)
ps = json.load(open("out/price_shape.json"))
pd.DataFrame([dict(period=ps[k]["period"], mae_usd=round(ps[k]["mae_usd"], 2), solar_capture_actual=round(ps[k]["solar_capture_actual"], 3),
                   solar_capture_model=round(ps[k]["solar_capture_model"], 3), zero_actual=ps[k]["zero_hours_actual"], zero_model=ps[k]["zero_hours_model"])
              for k in ["train", "test"]]).to_csv("dash/price_shape.csv", index=False)
print(sorted(os.listdir("dash")))
