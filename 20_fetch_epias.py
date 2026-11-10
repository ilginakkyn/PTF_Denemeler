"""EPİAŞ Şeffaflık Platformu'ndan saatlik seriler (2016-01 → bugün), aylık parçalarla.
Kimlik: yalnızca oturum anahtarı (TGT) dosyası okunur; kullanıcı adı/şifre bu kodda yoktur.
Kullanım: python 20_fetch_epias.py <tgt_dosyası>
"""
import sys, json, time, os, urllib.request
import pandas as pd

TGT = open(sys.argv[1]).read().strip()
BASE = "https://seffaflik.epias.com.tr/electricity-service/v1/"
OUT = "/home/claude/proj/data/epias"
os.makedirs(OUT, exist_ok=True)
SERIES = {
    "ptf": "markets/dam/data/mcp",
    "smf": "markets/bpm/data/system-marginal-price",
    "direction": "markets/bpm/data/system-direction",
    "generation": "generation/data/realtime-generation",
    "consumption": "consumption/data/realtime-consumption",
    "dam_fullness": "dams/data/active-fullness",
}
START = pd.Timestamp("2016-01-01"); END = pd.Timestamp("2026-10-28")

def call(ep, s, e):
    body = json.dumps({"startDate": s.strftime("%Y-%m-%dT00:00:00+03:00"),
                       "endDate": e.strftime("%Y-%m-%dT00:00:00+03:00")}).encode()
    for k in range(4):
        try:
            req = urllib.request.Request(BASE + ep, data=body, method="POST",
                                         headers={"Content-Type": "application/json", "TGT": TGT})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r).get("items", [])
        except Exception as ex:
            print(f"  retry {ep} {s.date()} ({ex})", flush=True); time.sleep(3 * (k + 1))
    raise RuntimeError(f"{ep} {s.date()} başarısız")

for name, ep in SERIES.items():
    path = f"{OUT}/{name}.parquet"
    if os.path.exists(path):
        print(name, "var, atlandı"); continue
    rows = []
    months = pd.date_range(START, END, freq="MS")
    for m0 in months:
        m1 = min(m0 + pd.offsets.MonthEnd(0), END)
        if name == "dam_fullness":          # günlük-baraj bazlı: ayın yalnız 1. ve 15. günü (boyut için)
            for d in (m0, m0 + pd.Timedelta(days=14)):
                rows += call(ep, d, d)
        else:
            rows += call(ep, m0, m1)
        print(f"{name} {m0:%Y-%m} toplam {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    print("YAZILDI", name, df.shape, flush=True)
print("TAMAM")
