#Otonom Araç Güvenlik Modülü — SVM Navigasyon

**Algoritma Analizi ve Tasarımı Dersi | Java 17**

Otonom bir araç navigasyon sisteminde, iki sınıf engel koordinatını **en geniş güvenlik koridoru** ile ayıran matematiksel modelin Java implementasyonu. Çözüm olarak **Hard-Margin Linear SVM** ve **SMO (Sequential Minimal Optimization)** algoritması kullanılmıştır.

---

## Matematiksel Model
İki sınıfı birbirinden ayıran, her iki sınıfa da eşit uzaklıktaki karar sınırını (`w·x + b = 0`) bulmayı hedefler.

* **Optimizasyon:** $\min \frac{1}{2} \|w\|^2$
* **Kısıt:** $y_i(w \cdot x_i + b) \geq 1$
* **Çözüm:** Lagrange Dual formu üzerinden SMO ile hesaplanır.

---

## Teknik Özellikler
* **Algoritma:** SMO (Sequential Minimal Optimization) - $O(n^2 \cdot iter)$
* **Mimari:** Strategy Pattern (Kernel yönetimi), Factory Pattern (Veri seti üretimi)
* **Bellek:** Kernel önbellekleme (Caching) ile $O(n^2)$ performans.
* **Prensipler:** Immutable veri yapıları, SOLID prensipleri ve hata yönetimi.

### Zaman Karmaşıklığı
| İşlem | Karmaşıklık |
| :--- | :--- |
| Kernel Cache | $O(n^2)$ |
| SMO Eğitim | $O(n^2 \cdot iter)$ |
| Tahmin (Inference) | $O(n_{sv})$ |

---

## Proje Yapısı
```text
src/svm/
├── model/           # DataPoint, SVMResult (Immutable)
├── kernel/          # KernelFunction (Strategy Pattern)
├── algorithm/       # SVMTrainer (SMO Logic)
├── reporter/        # Çıktı formatlama
└── app/             # Main (Orkestrasyon)
