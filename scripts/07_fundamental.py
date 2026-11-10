"""Temel (fundamental) saatlik piyasa modeli — merit-order + hidro + depolama + ticaret.

Her gün için bir doğrusal program (LP) çözülür:
  min  Σ_t Σ_k MC_k · g_k,t  + VOLL · açık_t + C_imp · ithalat_t − V_exp · ihracat_t
  s.t. Σ_k g_k,t + hidro_baraj_t + deşarj_t − şarj_t + ithalat_t − ihracat_t − kısıntı_t + açık_t
         = Talep_t − (GES_t + RES_t + akarsu_t + nükleer + jeotermal + biyokütle + kömür_minimum)
       Σ_t hidro_baraj_t = günlük su bütçesi, 0 ≤ hidro_baraj_t ≤ kurulu baraj gücü
       depolama SoC dinamiği, gün başı = gün sonu = %50
Saatlik fiyat = denge kısıtının dual değeri (marjinal maliyet), [0, tavan] aralığında.

Şekiller (2025 hava yılı, gerçek veri): saatlik tüketim (EPİAŞ), lisanslı GES ve rüzgâr üretimi
(kapasite ile normalize). Senaryolar bu şekilleri ölçekler.
"""
import numpy as np, pandas as pd, json, sys
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

D = pd.read_parquet("data/tr_hourly.parquet")
Y = D.loc["2025-01-01":"2025-12-31 23:00"]
H = len(Y)

# ---------------- şekiller ----------------
load_shape = Y["consumption_mwh"].to_numpy() / Y["consumption_mwh"].sum()      # toplam = 1

def cf_from(actual, cap_guess_start, cap_guess_end):
    cap = np.linspace(cap_guess_start, cap_guess_end, len(actual))
    return np.clip(actual / cap, 0, 1)

# Lisanslı GES: kapasite vekili = 30 günlük kayan max / 0.85 (bkz. 06_aggregator)
sol = D.loc["2024-12-01":"2025-12-31 23:00", "act_solar"].astype(float)
solcap = (sol.rolling(24 * 30, min_periods=24 * 7).max().bfill() / 0.85).loc["2025"]
solar_cf_raw = (sol.loc["2025"] / solcap).clip(0, 1).to_numpy()
wind = Y["act_wind"].astype(float).to_numpy()
# Rüzgâr kapasitesi 2025 başı ~13.0 GW, sonu 14.77 GW (ETKB) — EPİAŞ gerçek zamanlı üretim
wind_cf_raw = cf_from(wind, 13000, 14774)

def scale_cf(cf, target_mean):
    """Şekli korur, ortalamayı hedefe getirir (1'i aşmadan)."""
    k = target_mean / cf.mean()
    for _ in range(20):
        c = np.clip(cf * k, 0, 1); k *= target_mean / c.mean()
    return np.clip(cf * k, 0, 1)

# hidro aylık dağılım (Türkiye tipik; DSİ/TEİAŞ uzun dönem ortalamasına yakın) — varsayım
# hidro aylık dağılım: EPİAŞ gerçek zamanlı üretim 2016-2025 ortalaması (baraj + akarsu)
_E = pd.read_parquet('data/epias_hourly.parquet').loc['2016':'2025']
_h = (_E.gen_dammedHydro + _E.gen_river).astype(float); _m = _h.groupby([_h.index.year, _h.index.month]).sum().unstack()
HYDRO_MONTH = np.array(_m.div(_m.sum(axis=1), axis=0).mean(), dtype=float); HYDRO_MONTH = HYDRO_MONTH / HYDRO_MONTH.sum()
month = Y.index.month.to_numpy() - 1
day_id = (Y.index.normalize() - Y.index[0].normalize()).days.to_numpy()

def build_supply_curve(fuel):
    """Termik kapasiteyi alt basamaklara bölerek pürüzsüz bir arz eğrisi kurar."""
    tiers = []
    def add(name, cap_mw, mc_lo, mc_hi, n=5):
        for i in range(n):
            tiers.append((name, cap_mw / n, mc_lo + (mc_hi - mc_lo) * (i + 0.5) / n))
    gas_th = fuel["gas_usd_mwh_th"]; co2 = fuel["co2_usd_t"]
    av = fuel["thermal_availability"]
    # yerli linyit (minimum yükün üstü)
    avL = fuel.get("lignite_availability", av); avC = fuel.get("coal_availability", av)
    add("linyit", fuel["lignite_mw"] * avL * (1 - fuel["lignite_min"]), 38 + 1.1 * co2, 55 + 1.1 * co2)
    # ithal kömür: $/t -> $/MWh_e ; 25.1 GJ/t (6000 kcal/kg), %38 verim
    coal_e = fuel["coal_usd_t"] / (25.1 / 3.6) / 0.38
    add("ithal_komur", fuel["coal_mw"] * avC * (1 - fuel["coal_min"]), coal_e * 0.9 + 3 + 0.9 * co2, coal_e * 1.1 + 5 + 0.9 * co2)
    add("dogalgaz_KÇ", fuel["ccgt_mw"] * av * (1 - fuel.get("gas_min", 0)), gas_th / 0.57 + 3 + 0.36 * co2, gas_th / 0.50 + 4 + 0.40 * co2)
    add("dogalgaz_eski", fuel["ocgt_mw"] * av, gas_th / 0.45 + 5 + 0.45 * co2, gas_th / 0.36 + 7 + 0.56 * co2)
    return tiers

def solve_year(s, return_detail=False):
    """s: senaryo sözlüğü. 8760 saatlik fiyat ve dağıtım döndürür."""
    load = load_shape * s["demand_twh"] * 1e6                         # MWh/h
    if s.get("ev_twh", 0) > 0:                                        # EV şarj profili: akşam-gece ağırlıklı
        hrs = Y.index.hour.to_numpy()
        prof = np.where((hrs >= 18) | (hrs <= 5), 1.6, np.where((hrs >= 10) & (hrs <= 15), 0.9, 0.6))
        load = load + prof / prof.sum() * s["ev_twh"] * 1e6
    solar = scale_cf(solar_cf_raw, s["solar_cf"]) * s["solar_mw"]
    windg = scale_cf(wind_cf_raw, s["wind_cf"]) * s["wind_mw"]
    hydro_month_twh = HYDRO_MONTH * s["hydro_twh"]
    days_in_month = pd.Series(Y.index.normalize()).groupby(Y.index.month).nunique().to_numpy()
    ror = np.array([hydro_month_twh[m] * s["ror_share"] * 1e6 / (days_in_month[m] * 24) for m in month])
    res_day_budget = {}
    for d in np.unique(day_id):
        m = month[day_id == d][0]
        res_day_budget[d] = hydro_month_twh[m] * (1 - s["ror_share"]) * 1e6 / days_in_month[m]
    res_cap = s["hydro_mw"] * s.get("res_power_factor", 0.9) - ror                               # baraj tipi kalan güç
    must = (solar + windg + ror + s["nuclear_mw"] * 0.88 + s["geo_mw"] * 0.72 + s["bio_mw"] * 0.62
            + s["lignite_mw"] * s.get("lignite_availability", s["thermal_availability"]) * s["lignite_min"]
            + s["coal_mw"] * s.get("coal_availability", s["thermal_availability"]) * s["coal_min"]
            + s["ccgt_mw"] * s["thermal_availability"] * s.get("gas_min", 0))
    resid = load - must
    vre = solar + windg                                               # kısıntı yalnız rüzgâr+güneşten
    tiers = build_supply_curve(s)
    K = len(tiers); mc = np.array([t[2] for t in tiers]); tcap = np.array([t[1] for t in tiers])
    P, E, rte = s["storage_mw"], s["storage_mwh"], 0.87
    ec = ed = np.sqrt(rte)
    price = np.zeros(H); curt = np.zeros(H); gen_k = np.zeros((H, K)); hres = np.zeros(H)
    stor_c = np.zeros(H); stor_d = np.zeros(H); exp_ = np.zeros(H); imp_ = np.zeros(H); uns = np.zeros(H)
    VOLL = s.get("voll", 3000.0)          # yük atma maliyeti; fiyat tavanı yalnız raporlanan fiyata uygulanır
    CAP = s["price_cap"]
    for d in np.unique(day_id):
        idx = np.where(day_id == d)[0]; n = len(idx)
        # değişken sırası: g(K*n), hres(n), c(n), dch(n), soc(n), curt(n), exp(n), imp(n), uns(n)
        nv = K * n + 8 * n
        o = lambda j: K * n + j * n
        cost = np.zeros(nv)
        for k in range(K):
            cost[k * n:(k + 1) * n] = mc[k]
        cost[o(1):o(2)] = 1e-3; cost[o(2):o(3)] = 0.5                 # şarj/deşarj küçük aşınma maliyeti
        cost[o(4):o(5)] = 1e-3                                        # kısıntı (fiyat ~0)
        cost[o(5):o(6)] = -s["export_value"]
        cost[o(6):o(7)] = s["import_cost"]
        cost[o(7):o(8)] = VOLL
        Aeq = lil_matrix((n + n + 2, nv)); beq = np.zeros(n + n + 2)
        for t in range(n):
            for k in range(K):
                Aeq[t, k * n + t] = 1
            Aeq[t, o(0) + t] = 1; Aeq[t, o(1) + t] = -1; Aeq[t, o(2) + t] = 1
            Aeq[t, o(4) + t] = -1; Aeq[t, o(5) + t] = -1; Aeq[t, o(6) + t] = 1; Aeq[t, o(7) + t] = 1
            beq[t] = resid[idx[t]]
            # SoC
            r = n + t
            Aeq[r, o(3) + t] = 1
            if t > 0:
                Aeq[r, o(3) + t - 1] = -1
            Aeq[r, o(1) + t] = -ec; Aeq[r, o(2) + t] = 1 / ed
            beq[r] = 0.5 * E if t == 0 else 0
        Aeq[2 * n, o(3) + n - 1] = 1; beq[2 * n] = 0.5 * E
        Aeq[2 * n + 1, o(0):o(0) + n] = 1; beq[2 * n + 1] = res_day_budget[d]
        b = []
        for k in range(K):
            b += [(0, tcap[k])] * n
        b += [(0, max(res_cap[i], 0)) for i in idx]
        b += [(0, P)] * n + [(0, P)] * n + [(0, E)] * n
        b += [(0, float(vre[i])) for i in idx] + [(0, s["export_mw"])] * n + [(0, s["import_mw"])] * n + [(0, None)] * n
        rr = linprog(cost, A_eq=csr_matrix(Aeq), b_eq=beq, bounds=b, method="highs")
        if rr.status != 0:
            # su bütçesi kapasiteye sığmıyorsa: bütçeyi kapasiteye kırp ve tekrar dene
            beq[2 * n + 1] = min(res_day_budget[d], sum(max(res_cap[i], 0) for i in idx) * 0.999)
            rr = linprog(cost, A_eq=csr_matrix(Aeq), b_eq=beq, bounds=b, method="highs")
        x = rr.x
        price[idx] = np.clip(rr.eqlin.marginals[:n], 0, CAP)
        for k in range(K):
            gen_k[idx, k] = x[k * n:(k + 1) * n]
        hres[idx] = x[o(0):o(1)]; stor_c[idx] = x[o(1):o(2)]; stor_d[idx] = x[o(2):o(3)]
        curt[idx] = x[o(4):o(5)]; exp_[idx] = x[o(5):o(6)]; imp_[idx] = x[o(6):o(7)]; uns[idx] = x[o(7):o(8)]
    # --- teklif davranışı / kıtlık katmanı (2025'e kalibre): fiyat = LP_maliyet × (a + b·u^γ)
    #     u = termik kullanım oranı (çalışan / emre amade termik). Maliyet-bazlı LP gün içi farkı
    #     olduğundan düşük gösterir; gerçek teklifler akşam sıkışıklığında prim, öğlen fazlasında
    #     fiyat-alıcı davranış içerir.
    therm_av = (s["lignite_mw"] + s["coal_mw"] + s["ccgt_mw"] + s["ocgt_mw"]) * s["thermal_availability"]
    therm_min = (s["lignite_mw"] * s["lignite_min"] + s["coal_mw"] * s["coal_min"] + s["ccgt_mw"] * s.get("gas_min", 0)) * s["thermal_availability"]
    u = np.clip((gen_k.sum(axis=1) + therm_min) / therm_av, 0, 1)
    mk = s.get("markup", None)
    price_lp = price.copy()
    if mk:
        price = np.clip(price_lp * (mk["a"] + mk["b"] * u ** mk["g"]), 0, CAP)
    fuels = sorted(set(t[0] for t in tiers))
    gen_f = {f: gen_k[:, [i for i, t in enumerate(tiers) if t[0] == f]].sum(axis=1) for f in fuels}
    gen_f["linyit"] += s["lignite_mw"] * s.get("lignite_availability", s["thermal_availability"]) * s["lignite_min"]
    gen_f["ithal_komur"] += s["coal_mw"] * s.get("coal_availability", s["thermal_availability"]) * s["coal_min"]
    gen_f["dogalgaz_KÇ"] = gen_f.get("dogalgaz_KÇ", 0) + s["ccgt_mw"] * s["thermal_availability"] * s.get("gas_min", 0)
    solar_used = solar - curt * np.where(solar + windg > 0, solar / np.maximum(solar + windg, 1e-9), 0)
    wind_used = windg - curt * np.where(solar + windg > 0, windg / np.maximum(solar + windg, 1e-9), 0)
    out = dict(
        avg_price=float(price.mean()),
        load_weighted_price=float((price * load).sum() / load.sum()),
        solar_capture=float((price * solar_used).sum() / max(solar_used.sum(), 1)),      # sevk edilen üretimle ağırlıklı
        wind_capture=float((price * wind_used).sum() / max(wind_used.sum(), 1)),
        solar_capture_potential=float((price * solar).sum() / max(solar.sum(), 1)),
        zero_price_hours=int((price <= 1.0).sum()),
        curtail_twh=float(curt.sum() / 1e6),
        curtail_share_vre=float(curt.sum() / max((solar + windg).sum(), 1)),
        solar_twh=float(solar.sum() / 1e6), wind_twh=float(windg.sum() / 1e6),
        vre_share_of_demand=float((solar_used + wind_used).sum() / load.sum()),
        hydro_twh=float((ror.sum() + hres.sum()) / 1e6),
        gas_twh=float((gen_f.get("dogalgaz_KÇ", 0) + gen_f.get("dogalgaz_eski", 0)).sum() / 1e6),
        coal_twh=float((gen_f["linyit"] + gen_f["ithal_komur"]).sum() / 1e6),
        lignite_twh=float(gen_f["linyit"].sum() / 1e6), import_coal_twh=float(gen_f["ithal_komur"].sum() / 1e6),
        export_twh=float(exp_.sum() / 1e6), import_twh=float(imp_.sum() / 1e6),
        unserved_mwh=float(uns.sum()),
        storage_throughput_twh=float(stor_d.sum() / 1e6),
        demand_twh=float(load.sum() / 1e6),
    )
    out["solar_capture_ratio"] = out["solar_capture"] / out["avg_price"] if out["avg_price"] > 0 else None
    out["wind_capture_ratio"] = out["wind_capture"] / out["avg_price"] if out["avg_price"] > 0 else None
    hp = pd.Series(price).groupby(Y.index.hour).mean()
    out["hour_profile"] = [round(v, 2) for v in hp.tolist()]
    out["month_avg"] = [round(v, 2) for v in pd.Series(price).groupby(Y.index.month).mean().tolist()]
    pr = pd.Series(price, index=Y.index)
    dd = pr.resample("D").apply(lambda x: np.sort(x.values)[-4:].mean() - np.sort(x.values)[:4].mean())
    out["spread4_avg"] = float(dd.mean())
    out["_u"] = u; out["_price_lp"] = price_lp
    if return_detail:
        return out, pr, pd.Series(solar, index=Y.index), pd.Series(windg, index=Y.index), pd.Series(load, index=Y.index)
    return out
