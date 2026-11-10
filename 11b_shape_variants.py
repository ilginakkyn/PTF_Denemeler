import json, numpy as np, pandas as pd
exec(open("scripts/11_price_shape.py").read().split("tr = (X.index")[0])
tr = (X.index >= "2024-07-01") & (X.index < "2026-01-01"); te = X.index >= "2026-01-01"
def design(Xd):
    H = pd.get_dummies(Xd["hour"], prefix="h").astype(float)
    M = pd.get_dummies(Xd["month"], prefix="m").astype(float)
    z = Xd["z"]
    return pd.concat([z.rename("z"), (z**2).rename("z2"), (Xd["dow"]>=5).astype(float).rename("we"), H, M], axis=1)
from sklearn.linear_model import Ridge, HuberRegressor
Xd = design(X)
for name, mdl in [("ridge_z2", Ridge(1.0)), ("huber_z2", HuberRegressor(max_iter=500, alpha=1e-3))]:
    mdl.fit(Xd[tr], y[tr]); rh = pd.Series(mdl.predict(Xd), index=X.index).clip(lower=0); ph = rh*pm
    for lab, mk in [("train", tr), ("test", te)]:
        pa, pp, s = p[mk], ph[mk], solar_tot[mk]
        print(name, lab, "MAE", round(float(np.abs(pa-pp).mean()),2), "cap", round(float((pp*s).sum()/s.sum()/pp.mean()),3), "act", round(float((pa*s).sum()/s.sum()/pa.mean()),3), "zero", int((pp<=1).sum()), "act0", int((pa<=1).sum()))
    if name=="ridge_z2": print("coef z, z2:", mdl.coef_[:3].round(3))
