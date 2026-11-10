# Türkiye VPP / Toplayıcılık ve PTF Simülasyonu — özet (güncelleme 29 Ekim 2026)

**Veri:** EPİAŞ Şeffaflık Platformu'ndan doğrudan, 2016-01 → 2026-10-28 saatlik: PTF, SMF, sistem yönü, kaynak bazlı üretim, tüketim. Yıllık TL ortalamaları 2019–2024 EPİAŞ raporlarıyla birebir.

## Bulgular
1. **Güneş kendi fiyatını düşürüyor:** 12:00 fiyatı/yıllık ort. 2016'da 1.14 (öğle zirveydi) → 2019 0.96 → 2026 0.49. Sıfır fiyatlı saat: 2025'te 87, 2026'da 28 Ekim'e kadar 449. Lisanslı GES yakalama oranı 2019: 1.02 → 2025: 0.83 → 2026: 0.75.
1b. **Dengesizlik pahalılaştı:** gerçek SMF/yönle birim bedel PTF'nin 2025'te %13 (fazla) / %10 (eksik), 2026 yeni formülle %22 / %25 (saatlik SMF ile alt sınır).
2. **PTF tahmini:** fiyat geçmişi + TEİAŞ yük tahmini + takvimle LightGBM/Ridge topluluğu; 2 yıllık walk-forward testte MAE 7.70 $/MWh (naif 10.67, %28 daha iyi); bağımsız açık model Puant ile eşit (7.69). Sızıntı testi temiz. Kalibre %80 aralık kapsaması 0.79. EPİAŞ hidro/gaz/DGP girdileri (dünün değerleri) iyileştirmedi (7.76).
3. **Toplayıcı:** 40 dağınık GES (7 bölge) dengesizlik hacmini baz varsayımla ~%37 azaltıyor (duyarlılık %21–47); tek bölgede ~%11. 10–20 tesisten sonra doyuyor — kalan ulusal hava tahmini hatası. Gerçek dengesizlik fiyatlarıyla tasarruf ~0.81 $/MWh üretim (gelirin %1.6'sı; aralık 0.36–1.34 $): tek başına iş modeli değil.
4. **Batarya:** yalnız arbitraj (2 saat, 1 MW) ~27.5 bin $/MW/yıl → 15 yılda geri ödemiyor. Gerçek EPİAŞ PFK fiyatlarıyla (son 12 ay ort. 41 $/MW-saat) PFK + arbitraj ~362 bin $/MW/yıl → ~2 yılda geri ödüyor; SFK eklenince ~571 bin. Ama PFK pazarı ~390 MW, SFK ~980 MW (2026 pazar: ~116 M$ + ~480 M$); lisanslı depolama projeleri bunu aşıyor → ilk giren avantajı, sonra fiyat düşer. Erişim mevzuata bağlı.
5. **2035:** 120 GW (resmi dağılım 76.9 güneş + 43.1 rüzgâr) + UEP talebi (510 TWh) + 7.5 GW/2s batarya, ortalama hidroloji (78.6 TWh) → GES yakalama 0.34, ~2,516 sıfır fiyatlı saat, ~12.5 TWh kısıntı (%5). 20 GW/4s depolamayla kısıntı ~4.3 TWh, yakalama 0.57. Kurak yılda (62 TWh hidro) ort. fiyat 43.3 $, yağışlı yılda (109 TWh) 34.0 $.
5b. **Senaryo yolları 2026–2035:** kötü (pahalı gaz 55 $/MWh-th, talep %4.7/yıl, 80 GW rüzgâr+güneş, az depolama), baz (resmî plan), iyi (ucuz gaz, düşük talep, 20 GW/4s depolama) × 2016–2025'in on hava yılı = 300 yıllık çözüm. 2030 PTF ortancası: kötü 102 $ (%10–90: 95–105), baz 51.5 $ (48–54), iyi 39.3 $ (36–42); 2035: 109 / 54 / 32 $. Sıfır fiyatlı saat 2030: 308 / 1,662 / 2,391. Kötü senaryoda yeni güvenilir kapasite eklenmediği için 2030'da yılda ~160 saat (0.29 TWh) talep karşılanamıyor; 2035'te ~450 saat (1.4 TWh). Bu bir kesinti öngörüsü değil, gereken kapasitenin ölçüsü.
5c. **Stres haftaları (2030):** rüzgârsız kış, sıcak hava dalgası (+%7), gaz arzı kesintisi (−%40), büyük santral kaybı (−4.4 GW), kurak yılda sıcak hafta. Baz sistemde kesinti yok, fiyat haftada 17–45 saat tavanda. Kötü sistemde tepe açık 9–22 GW (tepe talebin %14–27'si), haftada 62–75 saat; +2 GW/4 GWh esneklik açığı yalnız %6–15 azaltıyor.
5d. **Geriye dönük kontrol:** model 2026'nın gerçek hidrolojisiyle Ocak–Ekim ort. PTF'yi 47.6 $ buluyor, gerçekleşen 44.5 $ (+%7). 2026'nın rekor hidrosu (~103 TWh) on hava yılının dışında; bantlar bu kadar yağışlı bir yılı içermiyor.
6. **Dış kaynaklarla karşılaştırma (APLUS Enerji):** Aralık 2025 sunumunda GES yakalama 2024 ~0.91 (bizim lisanslı GES: 0.91). Energy Sector Report 2026: 2028 batarya hedefi 10 GW, Nisan 2026'da 36 toplayıcı lisansı, 15 dk dengesizlik uzlaştırması 2027 hedefli → PFK/SFK pazarının doyması büyük olasılık.

## Yeniden üretilebilirlik
Temiz klasörde `run_all.sh` ile baştan üretildi: 19 sonuç dosyasının tamamı bayt bayt aynı. Ana tablo artık yalnız EPİAŞ + lisanslı kaynaklardan (Puant MIT, Çetin–Sağlam–Oğuz CC BY 4.0). Colab defteri hatasız çalışıyor.

## Sınırlar
Fiyat modeli hava/hidroloji kullanmıyor; piyasa modelinde akarsu payı (%75) sabit, esnek değil; senaryo bantları 2026 gibi aşırı yağışlı yılları içermiyor; 2035 modeli maliyet bazlı (2025'te gün içi farkı yarı yarıya düşük gösterdi → gerçek yamyamlık daha sert olabilir); 15 dakikalık SMF yok (2026 bedeli alt sınır); toplayıcıdaki yerel hata parametreleri varsayım.

## Sonraki adımlar
15 dakikalık SMF; fiyat modeline gün öncesi hava tahmini; gerçek tesis sayaç verisiyle toplayıcı kalibrasyonu; 2035 modeline rampa/elektrolizör/talep esnekliği.
