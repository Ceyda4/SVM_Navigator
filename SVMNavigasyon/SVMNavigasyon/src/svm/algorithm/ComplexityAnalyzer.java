package svm.algorithm;

import svm.model.SVMResult;
public final class ComplexityAnalyzer {

    /** Utility class — instance oluşturulamaz */
    private ComplexityAnalyzer() {}

    /**
     * Verilen parametreler için teorik işlem sayısını döndürür.
     *
     *  ADIM ADIM ANALİZ:
     *  ─────────────────
     *  1. KernelCache (ön-hesaplama):
     *     n² kernel hesabı → O(n²)
     *     Her K(i,j) = 2 çarpma + 1 toplama → 3 işlem
     *     Toplam: 3n² işlem
     *
     *  2. SMO dış döngü: maxIter kez
     *  3. SMO iç döngü (i): n nokta taranır
     *  4. f(xᵢ) hesabı (önbellek ile): O(1) lookup (önbelleksiz O(n))
     *  5. j seçimi: O(n)
     *  6. Güncelleme: O(1) — analitik formül
     *  7. Hata güncelleme (recompute): O(n²) — yalnızca zaman zaman
     *
     *  Dominant term:   O(n² × maxIter)
     *  Erken yakınsama: pratikte çok daha az
     *
     * @param n        nokta sayısı
     * @param maxIter  maksimum iterasyon
     * @return teorik üst sınır işlem sayısı
     */
    public static long theoreticalOperations(int n, int maxIter) {
        // Kernel cache: n²
        // Her iterasyon: n (tarama) × n (j seçimi) = n²
        // maxIter iterasyon: maxIter × n²
        return (long) n * n * (1 + maxIter);
    }

    /**
     * Tahmin aşaması karmaşıklığı.
     * f(x) = Σ(αᵢ≠0) αᵢyᵢK(xᵢ,x) + b
     * Zaman: O(n_sv × d)
     *
     * @param nSupportVectors support vector sayısı
     * @param dimension       uzay boyutu (2D için 2)
     * @return tahmin başına işlem sayısı
     */
    public static long predictionOperations(int nSupportVectors, int dimension) {
        return (long) nSupportVectors * dimension;
    }

    /**
     * Kernel cache'in megabyte cinsinden bellek kullanımı.
     * n×n double matris: n² × 8 byte
     *
     * @param n nokta sayısı
     * @return bellek kullanımı (MB)
     */
    public static double cacheMemoryMB(int n) {
        return ((long) n * n * Double.BYTES) / (1024.0 * 1024.0);
    }

    /**
     * Tüm karmaşıklık analizini biçimlendirilmiş metin olarak döndürür.
     *
     * @param n        nokta sayısı
     * @param maxIter  maksimum iterasyon
     * @param result   eğitim sonucu
     * @param elapsedMs ölçülen süre (milisaniye)
     * @return biçimlendirilmiş analiz raporu
     */
    public static String buildReport(int n, int maxIter,
                                      SVMResult result, double elapsedMs) {
        int    nSV           = result.getSupportVectors().size();
        long   theoritical   = theoreticalOperations(n, maxIter);
        long   predOps       = predictionOperations(nSV, 2);
        double memMB         = cacheMemoryMB(n);

        return String.join(System.lineSeparator(),
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║           ZAMAN KARMAŞİKLİĞİ ANALİZİ                       ║",
            "╠══════════════════════════════════════════════════════════════╣",
            String.format("║  Veri boyutu (n)       : %-36d║", n),
            String.format("║  Maks. iterasyon       : %-36d║", maxIter),
            String.format("║  Gerçek yakınsama      : %-36d║", result.getConvergedAt()),
            "╠══════════════════════════════════════════════════════════════╣",
            "║  EĞITIM (SMO)                                                ║",
            "║  ─────────────────────────────────────────────────          ║",
            "║  KernelCache ön-hesaplama : O(n²)                           ║",
            String.format("║    → n²  = %-4d işlem                                       ║",
                (long) n * n),
            "║  Ana döngü               : O(n² × maxIter)                  ║",
            String.format("║    → n²×iter = %-8d işlem (teorik üst sınır)           ║",
                (long) n * n * maxIter),
            String.format("║    → Toplam  : %-10d işlem                              ║",
                theoritical),
            String.format("║    → Gerçek süre: %-6.3f ms                                ║",
                elapsedMs),
            "╠══════════════════════════════════════════════════════════════╣",
            "║  TAHMİN (eğitim sonrası)                                     ║",
            "║  ─────────────────────────────────────────────────          ║",
            String.format("║  Support vector sayısı : %-36d║", nSV),
            String.format("║  Tahmin karmaşıklığı   : O(n_sv × d) = O(%d × 2) = O(%d)   ║",
                nSV, nSV * 2),
            String.format("║    → Her tahmin: %-5d çarpma + toplama                     ║",
                predOps),
            "╠══════════════════════════════════════════════════════════════╣",
            "║  ALAN KARMAŞİKLİĞİ                                           ║",
            "║  ─────────────────────────────────────────────────          ║",
            String.format("║  alpha dizisi          : O(n) = %d double                   ║", n),
            String.format("║  Hata önbelleği        : O(n) = %d double                   ║", n),
            String.format("║  Kernel matrisi        : O(n²) = %d double                  ║",
                (long) n * n),
            String.format("║    → ≈ %.4f MB                                              ║",
                memMB),
            String.format("║  Toplam bellek         : O(n²) dominant                     ║"),
            "╠══════════════════════════════════════════════════════════════╣",
            "║  KARŞILAŞTIRMA: Alternatif yöntemler                         ║",
            "║  ─────────────────────────────────────────────────          ║",
            "║  Naive QP (interior point) : O(n³)  — çok yavaş            ║",
            "║  SMO (bu implementasyon)   : O(n²×iter)  — verimli         ║",
            "║  Gradient Descent          : O(n×iter)  — yaklaşık         ║",
            "║  Chunking (Osuna 1997)     : O(n²)  — SMO öncesi           ║",
            "╚══════════════════════════════════════════════════════════════╝"
        );
    }
}
