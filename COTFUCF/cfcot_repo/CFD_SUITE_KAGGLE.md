# CF makalesi — kalan bütün koşular (suite)

GPU: **T4 x2**. Her tohum bir oturum (~7 saat), toplam üç oturum.

## Oturum 1

1. Notebook ayarı: Accelerator = GPU T4 x2, Internet = on.
2. `cfd_all.py`'nin tamamını bir hücreye yapıştır.
3. Sağ üstten **Save Version → Save & Run All (Commit)**.

Yapıştırılan hücre kendiliğinden `suite`'i başlatır: kontroller, ön kayıt,
tohum 10'un eğitimleri, kısayol görevi seçimi, sweep'ler. 11. saatte yeni iş
başlatmayı keser, kararı yazar ve `cfd_runs/suite_results.zip` olarak kaydeder.

Commit modu önemli: interaktif oturum kapanınca `/kaggle/working` silinir,
commit edilmiş sürümün çıktısı ise saklanır.

## Oturum 2 ve 3

1. Aynı notebook'ta **Add Input → Your Work → önceki sürümün çıktısı**.
2. Hücre aynı kalsın, yine **Save & Run All (Commit)**.

Süit `/kaggle/input` altındaki zip'i bulur, sonuçları geri yükler, bitmiş
tohumları atlar ve bir sonrakiyle devam eder. Zip'te checkpoint yoktur;
gerekmez.

## Bitince

Son sürümün çıktısındaki `suite_results.zip` içinde:

- `suite/prereg.json` — ön kayıt (ilk oturumda, veriden önce yazıldı)
- `suite/regime.json` — kısayol görevi seçimi
- `suite/verdict.md` — okunabilir karar
- `suite/runs/*/sweep*.jsonl` — bütün checkpoint ölçümleri

## Kayıtlı öngörüler

| | iddia | kural |
|---|---|---|
| T1 | unutma gerçekleşti, kontroller bastırdı | ood_cot NLL kayması ≥ 0.15; kontrolün kayması ≤ yarısı |
| **P1** | ood_cot, zincirli replay'den daha zincir-bağımlı | her tohumda 4 ölçünün ≥ 3'ü öngörülen yönde |
| **P2** | ood_cot, düz metin replay'den daha zincir-bağımlı | aynı kural |
| R1 | kısayol görevinde bağımlılık düşer (err_prop) | her tohumda DiD < 0, ortalama ≤ −0.05 |
| R2 | aynısı, cot_lift ile | aynı kural |

Ölçü hesaplanamazsa UNDEFINED, unutma olmadıysa UNINTERPRETABLE, taban
değeri çok düşükse FLOOR. Hiçbiri FAIL sayılmaz.

## Durdurmak

Hücreyi durdurmak çalışan eğitim ve sweep'leri de durdurur. Tekrar
çalıştırınca kaldığı yerden devam eder.

## Sadece yüklemek, çalıştırmamak

Yapıştırmadan önce ayrı hücrede: `%env CFD_NO_AUTORUN=1`
