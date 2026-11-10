# Türkiye VPP / Toplayıcılık ve PTF Simülasyonu — özet (28 Ekim 2026)

**Veri:** EPİAŞ saatlik PTF 2019-01 → 2026-08 (USD/MWh), tüketim 2016–2025, rüzgâr/lisanslı GES 2024-07 → 2026-10 (açık kaynak repolar üzerinden). Resmi yıllık/aylık ortalamalarla birebir doğrulandı.

## Bulgular
1. **Güneş kendi fiyatını düşürüyor:** 12:00 fiyatı/yıllık ort. 2019'da 0.96 → 2026'da 0.51. Sıfır fiyatlı saat: 2025'te 87, 2026 Oca–Ağu 427. Lisanslı GES yakalama oranı 2024: 0.94 → 2026: 0.73 (kısmi yıllar).
2. **PTF tahmini:** fiyat geçmişi + TEİAŞ yük tahmini + takvimle LightGBM/Ridge topluluğu; 2 yıllık walk-forward testte MAE 7.70 $/MWh (naif 10.67, %28 daha iyi); bağımsız açık model Puant ile eşit (7.69). Sızıntı testi temiz. Kalibre %80 aralık kapsaması 0.79.
3. **Toplayıcı:** 40 dağınık GES (7 bölge) dengesizlik hacmini baz varsayımla ~%37 azaltıyor (duyarlılık %21–47); tek bölgede ~%11. 10–20 tesisten sonra doyuyor — kalan ulusal hava tahmini hatası. Parasal tasarruf baz varsayımda 0.3–0.9 $/MWh üretim (gelirin ~%0.5–1.5'i): tek başına iş modeli değil.
4. **Batarya:** 2 saatlik 1 MW, bizim tahminle mükemmel öngörünün %89'u; son 12 ayda ~28.4 bin $/MW/yıl. 180 $/kWh capex ile 15 yıllık ömürde geri ödemiyor; başabaş için ~34 bin $/MW/yıl ek (yan hizmet/kapasite) gelir veya ~77 $/kWh capex gerekli.
5. **2035:** 120 GW rüzgâr+güneş + UEP talebi (510 TWh) + 7.5 GW/2s batarya → GES yakalama 0.37, ~2,400 sıfır fiyatlı saat, ~26 TWh kısıntı (%10). 20 GW/4s depolamayla kısıntı ~11 TWh, yakalama 0.59.

## Sınırlar
Fiyat modeli hava/hidroloji kullanmıyor; 2035 modeli maliyet bazlı (2025'te gün içi farkı yarı yarıya düşük gösterdi → gerçek yamyamlık daha sert olabilir); SMF verisi yok; toplayıcıdaki yerel hata parametreleri varsayım.

## Sonraki adımlar
EPİAŞ hesabıyla SMF, kaynak bazlı üretim, baraj doluluğu; fiyat modeline hava tahmini; gerçek tesis sayaç verisiyle toplayıcı kalibrasyonu; 2035 modeline rampa/elektrolizör/talep esnekliği.
