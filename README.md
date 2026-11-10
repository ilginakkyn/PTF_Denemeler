# PTF_Denemeler
# Türkiye VPP / PTF simülasyonu — kod ve veri

Bu klasör, panodaki tüm sayıları üreten kodu ve birleştirilmiş veriyi içerir. Her betik
proje kök dizininden çalıştırılır (`python scripts/<ad>.py`). Gerekenler: Python 3.11+,
`pandas numpy scikit-learn lightgbm scipy highspy pyarrow`.

## Veri
`data/tr_hourly.parquet` — saatlik, 2016-01-01 … 2026-10:
- `ptf_usd`, `load_plan_mw`, `kgup_mw` (2019-01 … 2026-08-27): EPİAŞ verisi, kaynak:
  github.com/beratkrts/electricity-price-forecasting-turkish-day-ahead-market (`tr_epf_ext.csv`)
- `consumption_mwh` + nüfus ağırlıklı sıcaklık/ışınım (2016–2025): github.com/ahalikoguz/turkiye-electricity-demand-forecasting
- `act_price` (TL), `act_consumption`, `act_wind`, `act_solar` (lisanslı) ve Puant tahminleri (2024-07 … 2026-10):
  github.com/emrtunali/puant (`data/backtest.parquet`)

Kendi EPİAŞ Şeffaflık hesabınızla (ücretsiz) bu seriler doğrudan çekilebilir; `eptr2` Python paketi işe yarar.

## Betikler (sırayla)
| Betik | Ne yapar |
|---|---|
| 01_build_data.py | Kaynakları tek saatlik tabloda birleştirir (yalnız ham repolar varsa gerekir) |
| 02_validate_data.py | Kaynaklar arası çapraz kontroller |
| 03_eda.py | Keşifsel analiz: profiller, sıfır fiyat saatleri, GES yakalama oranı |
| 04_ptf_model.py | Gün öncesi PTF tahmini, walk-forward backtest (≈20 dk) |
| 04b_eval.py, 04c_conformal.py | Hata metrikleri, kantil aralığı kalibrasyonu |
| 05_battery.py | Batarya arbitrajı (günlük LP) |
| 06_aggregator.py | Toplayıcı portföy dengesizlik simülasyonu |
| 07_fundamental.py | Saatlik maliyet bazlı piyasa modeli (kütüphane) |
| 08/09/13_*.py | 2025 kalibrasyonu |
| 11_price_shape.py | Ampirik fiyat-şekil modeli, 2026 dış örneklem testi |
| 12_scenarios_2035.py | 2035 senaryo ızgarası |
| 14_battery_econ.py | Batarya yatırım ekonomisi |
| 15_export_dash.py | Pano veri dosyaları |

Parametreler: `scripts/params_2025.json`, `scripts/calibration_choice.json`, `scripts/params_2035.json`
(kaynak notlarıyla). Sonuçlar `out/` klasöründe.
