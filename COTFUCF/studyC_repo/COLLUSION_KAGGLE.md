# Kaggle'da çalıştırma

GPU: T4 x1 yeter. Eğitim yok, sadece çıkarım.

## 1. Yapıştır

`col_all.py`'nin tamamını bir hücreye yapıştır ve çalıştır. Kendini
`/kaggle/working/col_all.py` olarak kaydeder, 48 kontrolü koşturur.

## 2. Tek satır

Yeni hücrede:

```
!python col_all.py suite
```

Sırayla yapar:

1. **Kontroller** — ölçüm makinesi, GPU harcamadan
2. **Kapı** — model oyunu oynayabiliyor mu. Geçmezse durur, hiçbir şey yazmaz
3. **Ön kayıt** — `prereg.json`, veri oluşmadan önce, zaman damgası ve kodun
   hash'i ile. Asla üzerine yazılmaz
4. **Menü testi** — `blind` kolu üç menü konumunda. Menünün ortasını mı seçiyor,
   yoksa gerçekten bir fiyat eğilimi mi var
5. **Doğrulama koşusu** — beş kol, beş **yeni** tohum (100-104), 60 tur
6. **Karar** — kayıttaki eşiklerle, kayıttaki öngörüler test edilir

Süre: yaklaşık **4 saat**.

## Oturum ölürse

Aynı satırı tekrar çalıştır. Biten koşuları atlar, yarım kalan satırı onarır,
kaldığı yerden devam eder. Kaydı değiştirmez. Ayarlar kayıttan farklıysa
durur — kayıt sonradan düzenlenmez.

## Bitince

`/kaggle/working/col_runs/suite_results.zip` dosyasını indir. İçinde
`prereg.json`, `arms.jsonl`, `menu.jsonl`, `verdict.md`, `verdict.json`.

## Kayıtlı öngörüler

| | iddia | kural |
|---|---|---|
| **P1** (birincil) | Zincirdeki sayıları bozmak ajanları ayrıştırır | 5/5 tohumda gap(corrupted) > gap(full), p ≤ 0.05, oran ≥ 2 |
| P2 | Bozuk zincir, zincirsizlikten kötü | ≥ 4/5 tohum |
| P3 | Satır sırası koordinasyonu taşımaz | ≥ 4/5 tohum |
| P4 | Zincir, fiyatları görmenin ötesinde senkron katar | ≥ 4/5 tohum |
| **M1** | `blind` seviyesi menünün ortasını izler | eğim ≥ 0.5 merkeze yönelme, ≤ 0.2 önyargı |

Fiyat seviyesi raporlanır ama test edilmez — keşif koşusunda iki modluydu.

## Başka model

```
!python col_all.py suite --model Qwen/Qwen2.5-3B-Instruct --out /kaggle/working/col_3b
```

Farklı model için farklı `--out`: her klasörün kendi kaydı olur.
