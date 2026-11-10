import pandas as pd, numpy as np, json
b = pd.read_parquet("out/ptf_backtest.parquet")
cols = ["naive_D", "naive_week", "naive_mix", "ridge", "lgb_unscaled", "lgb_q50", "ensemble", "puant_ext"]
def met(d, c):
    e = d[c] - d["actual"]; a = d["actual"]
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()),
                sMAPE=(2*np.abs(e)/(np.abs(a)+np.abs(d[c])+1e-9)).mean()*100,
                rMAE=np.abs(e).mean()/np.abs(d["naive_D"]-a).mean())
common = b.dropna(subset=cols)
res = {"all_hours_n": len(common), "overall": {c: {k: round(v, 3) for k, v in met(common, c).items()} for c in cols}}
print(pd.DataFrame(res["overall"]).T.round(2))
per = {}
for lab, m in [("Eyl24-Ağu25", common.index < "2025-09-01"), ("Eyl25-Ağu26", common.index >= "2025-09-01")]:
    per[lab] = {c: round(met(common[m], c)["MAE"], 3) for c in cols}
res["by_period_MAE"] = per; print(pd.DataFrame(per).round(2))
# TL karşılığı MAE (Puant raporuyla kıyas için)
tl = {c: float((np.abs(common[c]-common["actual"])*common["fx_usdtry"]).mean()) for c in cols}
res["MAE_TL"] = {k: round(v, 1) for k, v in tl.items()}; print("TL MAE:", res["MAE_TL"])
# kantil kapsama
cov = ((b["actual"] >= b["lgb_q10"]) & (b["actual"] <= b["lgb_q90"])).mean()
res["q10_q90_coverage"] = round(float(cov), 3); print("80% band kapsama:", cov)
# saate göre MAE
res["mae_by_hour_ens"] = common.assign(e=np.abs(common.ensemble-common.actual)).groupby(common.index.hour).e.mean().round(2).tolist()
res["mae_by_hour_naive"] = common.assign(e=np.abs(common.naive_D-common.actual)).groupby(common.index.hour).e.mean().round(2).tolist()
# aylık MAE
res["mae_by_month"] = {str(k): v for k, v in common.assign(e=np.abs(common.ensemble-common.actual), n=np.abs(common.naive_D-common.actual)).resample("MS")[["e","n"]].mean().round(2).to_dict("index").items()}
imp = pd.read_csv("out/feature_importance.csv", index_col=0).iloc[:, 0]
res["importance_top"] = (imp/imp.sum()).round(4).head(15).to_dict(); print((imp/imp.sum()).round(3).head(12))
json.dump(res, open("out/ptf_eval.json", "w"), indent=1, default=str)
