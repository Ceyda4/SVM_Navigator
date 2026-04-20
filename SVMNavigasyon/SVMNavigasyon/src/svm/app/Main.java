package svm.app;

import svm.algorithm.ComplexityAnalyzer;
import svm.algorithm.SVMTrainer;
import svm.exception.SVMException;
import svm.kernel.LinearKernel;
import svm.model.DataPoint;
import svm.model.SVMResult;
import svm.reporter.SVMReporter;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Application (Uygulama Katmanı)
 *  SINIF  : Main — uygulama giriş noktası
 * ═══════════════════════════════════════════════════════════════════
 *
 *  KATMAN MİMARİSİ:
 *  ─────────────────────────────────────────────────────────────────
 *
 *    ┌─────────────────────────────────────────────────────────┐
 *    │  APPLICATION LAYER  (svm.app)                           │
 *    │   Main, DatasetFactory                                  │
 *    │   Orkestrasyon — katmanları bir araya getirir           │
 *    ├─────────────────────────────────────────────────────────┤
 *    │  REPORTER LAYER  (svm.reporter)                         │
 *    │   SVMReporter                                           │
 *    │   Çıktı formatlama — algoritma kodundan tamamen ayrı    │
 *    ├─────────────────────────────────────────────────────────┤
 *    │  ALGORITHM LAYER  (svm.algorithm)                       │
 *    │   SVMTrainer, ComplexityAnalyzer                        │
 *    │   Çekirdek SMO — veri veya sunumdan habersiz            │
 *    ├─────────────────────────────────────────────────────────┤
 *    │  KERNEL LAYER  (svm.kernel)                             │
 *    │   KernelFunction (interface), LinearKernel, KernelCache │
 *    │   Strategy Pattern — algoritmadan bağımsız              │
 *    ├─────────────────────────────────────────────────────────┤
 *    │  MODEL LAYER  (svm.model)                               │
 *    │   DataPoint, SVMResult                                  │
 *    │   Immutable veri nesneleri — saf veri, sıfır mantık     │
 *    ├─────────────────────────────────────────────────────────┤
 *    │  EXCEPTION LAYER  (svm.exception)                       │
 *    │   SVMException                                          │
 *    │   Domain-specific hata yönetimi                         │
 *    └─────────────────────────────────────────────────────────┘
 *
 *  OOP PRENSİPLERİ (Genel):
 *    • Single Responsibility: her sınıfın tek bir sorumluluğu var
 *    • Open/Closed: yeni kernel veya veri seti için var olan kod
 *                  değiştirilmez, yeni sınıf eklenir
 *    • Liskov Substitution: LinearKernel → KernelFunction yerine geçer
 *    • Interface Segregation: KernelFunction tek metod → şişirilmemiş
 *    • Dependency Inversion: SVMTrainer, KernelFunction arayüzüne bağlı
 *
 *  BELLEK YÖNETİMİ:
 *    • DataPoint: immutable, GC-friendly
 *    • SVMResult: defensive copy ile dışarıdan mutasyona kapalı
 *    • KernelCache: train() biter bitmez clear() ile GC'ye bırakılır
 *    • Hiçbir static mutable state yok → bellek sızıntısı riski sıfır
 *    • try-catch bloğu → SVMException'da yarım kalan nesneler GC'ye kalır
 */
public final class Main {

    public static void main(String[] args) {

        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║     OTONOM ARAÇ GÜVENLİK MODÜLÜ — SVM NAVIGASYON            ║");
        System.out.println("║     Algoritma Analizi ve Tasarımı Ödevi                      ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");

        try {
            // ── 1. VERİ HAZIRLAMA ────────────────────────────────────────
            // Factory Pattern: veri üretimi Main'den ayrıştırıldı
            DataPoint[] trainingData = DatasetFactory.createObstacleDataset();

            System.out.printf("%nVeri seti yüklendi: %d nokta%n", trainingData.length);

            // ── 2. MODEL YAPILANDIRMA ────────────────────────────────────
            // Dependency Injection: LinearKernel dışarıdan verilir
            // SVMTrainer'ı değiştirmeden farklı kernel kullanılabilir
            SVMTrainer trainer = new SVMTrainer(
                LinearKernel.INSTANCE   // Kernel: K(a,b) = a·b
                // Varsayılanlar: C=1e10, maxIter=500, tol=1e-4, eps=1e-5
            );

            // ── 3. EĞİTİM ───────────────────────────────────────────────
            System.out.println("SMO algoritması çalıştırılıyor...");

            long startTime = System.nanoTime();
            SVMResult result = trainer.train(trainingData);
            long endTime   = System.nanoTime();

            double elapsedMs = (endTime - startTime) / 1_000_000.0;
            System.out.printf("Eğitim tamamlandı: %.3f ms  (yakınsama: %d. iterasyon)%n",
                elapsedMs, result.getConvergedAt());

            // ── 4. RAPOR ─────────────────────────────────────────────────
            SVMReporter.printFullReport(trainingData, result, elapsedMs);

            // ── 5. YENİ NOKTA TAHMİNİ ────────────────────────────────────
            System.out.println();
            System.out.println("── [7] YENİ ENGEL TAHMİNİ ─────────────────────────────────");
            System.out.println("   (Araç bu koordinatlarda hangi engel sınıfıyla karşılaşır?)");
            System.out.println();

            double[][] testPoints = {
                {1.5, 2.5},   // Sınıf A bölgesinde bekleniyor
                {7.0, 5.5},   // Sınıf B bölgesinde bekleniyor
                {4.5, 4.0},   // Ortada — sınır yakını
                {0.0, 0.0},   // Köşe — kesin Sınıf A
                {9.0, 8.0},   // Köşe — kesin Sınıf B
            };

            for (double[] tp : testPoints) {
                SVMReporter.printPrediction(tp[0], tp[1], result);
            }

            // ── 6. KARMAŞİKLİK ANALİZİ ──────────────────────────────────
            System.out.println(ComplexityAnalyzer.buildReport(
                trainingData.length, 500, result, elapsedMs
            ));

        } catch (SVMException e) {
            // Domain hataları: geçersiz veri, ayrılamayan sınıflar vb.
            System.err.println("[HATA] SVM hatası: " + e.getMessage());
            if (e.getCause() != null) {
                System.err.println("  Köken: " + e.getCause().getMessage());
            }
            System.exit(1);

        } catch (Exception e) {
            // Beklenmedik hatalar
            System.err.println("[BEKLENMEDIK HATA] " + e.getClass().getSimpleName() +
                               ": " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(2);
        }
    }
}
