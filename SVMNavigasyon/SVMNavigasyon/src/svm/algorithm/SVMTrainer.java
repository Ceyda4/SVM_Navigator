package svm.algorithm;

import svm.exception.SVMException;
import svm.kernel.KernelCache;
import svm.kernel.KernelFunction;
import svm.model.DataPoint;
import svm.model.SVMResult;

import java.util.ArrayList;
import java.util.List;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Algorithm (Çekirdek Algoritma Katmanı)
 *  SINIF  : SVMTrainer — SMO algoritmasını çalıştıran eğitici
 * ═══════════════════════════════════════════════════════════════════
 *
 *  ──────────────────────────────────────────────────────────────────
 *  ALGORİTMA: Sequential Minimal Optimization (SMO)
 *  KAYNAK:    John Platt, "Fast Training of Support Vector Machines
 *             Using Sequential Minimal Optimization", 1998
 *  ──────────────────────────────────────────────────────────────────
 *
 *  MATEMATİK TEMELİ:
 *  ─────────────────
 *  Primal problem:
 *    Minimize   ½‖w‖²
 *    Subject to yᵢ(w·xᵢ + b) ≥ 1  ∀i
 *
 *  Lagrange dual:
 *    Maximize   Σαᵢ − ½ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
 *    Subject to Σαᵢyᵢ = 0,  0 ≤ αᵢ ≤ C
 *
 *  KKT Koşulları (optimalite şartları):
 *    αᵢ = 0   →  yᵢ·f(xᵢ) ≥ 1   (margin dışında)
 *    0<αᵢ<C  →  yᵢ·f(xᵢ) = 1   (support vector — tam sınırda)
 *    αᵢ = C   →  yᵢ·f(xᵢ) ≤ 1   (slack aktif)
 *
 *  SMO Ana Fikri:
 *    Her adımda en az 2 alpha seçilmeli (Σαᵢyᵢ = 0 kısıtı).
 *    2 alpha'nın optimal değeri KAPALI FORM ile bulunur:
 *      αⱼ_new = αⱼ + yⱼ(Eᵢ − Eⱼ) / η   sonra kırpılır [L, H]
 *      αᵢ_new = αᵢ + yᵢyⱼ(αⱼ_old − αⱼ_new)
 *
 *  ──────────────────────────────────────────────────────────────────
 *  ZAMAN KARMAŞIKLİĞİ ANALİZİ:
 *  ─────────────────────────────
 *
 *  Ön-hesaplama (KernelCache):
 *    T_cache = O(n²)    — tüm K(i,j) değerleri bir kez hesaplanır
 *    A_cache = O(n²)    — n×n double matris
 *
 *  SMO ana döngüsü:
 *    Dış döngü:     O(MAX_ITER)   iterasyon
 *    İç döngü (i):  O(n)          tüm noktalar taranır
 *    f(i) hesabı:   O(n)          Σαᵢyᵢ K(i,·)
 *    j seçimi:      O(n)          max|Eᵢ−Eⱼ| heuristic
 *    Güncelleme:    O(1)          analitik formül
 *
 *    Toplam: O(n² × MAX_ITER)
 *    Ancak cache sayesinde f() içindeki her K(i,j): O(1)
 *    Pratik kompleksite: O(n × MAX_ITER) erken yakınsama ile
 *
 *  Sonuç işleme:
 *    w hesabı:   O(n)   — Σαᵢyᵢxᵢ
 *    SV tespiti: O(n)   — αᵢ > EPS filtresi
 *
 *  Tahmin (eğitim sonrası):
 *    O(n_sv)  — yalnızca support vectorlar kullanılır
 *    n_sv << n genellikle → çok hızlı
 *
 *  ──────────────────────────────────────────────────────────────────
 *  ALAN KARMAŞIKLİĞİ:
 *    alpha dizisi:   O(n)
 *    hata önbelleği: O(n)
 *    kernel önbelleği: O(n²)   ← dominant
 *    Toplam: O(n²)
 *  ──────────────────────────────────────────────────────────────────
 *
 *  OOP PRENSİPLERİ:
 *    • Single Responsibility: yalnızca SMO algoritmasını çalıştırır
 *    • Dependency Injection: KernelFunction dışarıdan verilir
 *    • Immutable config: final alanlar, builder pattern için hazır
 *    • Encapsulation: alpha, bias iç durumu dışa açılmaz;
 *      yalnızca SVMResult nesnesi döndürülür
 *
 *  BELLEK YÖNETİMİ:
 *    • alpha[] ve errorCache[] yalnızca train() ömrü boyunca yaşar
 *    • KernelCache.clear() çağrıldığında matris GC'ye bırakılır
 *    • Yerel double değişkenler stack'te → heap baskısı yok
 *    • SVMResult'a yalnızca kopyalar aktarılır → iç dizi sızmaz
 */
public final class SVMTrainer {

    // ─── Konfigürasyon sabitleri ───────────────────────────────────────────

    /**
     * Düzenleme parametresi C (hard-margin için çok büyük seçilir).
     *
     * C → ∞  : hard-margin SVM (hiç hata yok, ancak ayrılamayan veriye takılır)
     * C küçük: soft-margin SVM (bazı noktalar margin ihlal edebilir)
     *
     */
    private final double C;

    /** SMO maksimum iterasyon sayısı. Erken yakınsama olursa döngü kesilir. */
    private final int maxIter;

    /**
     * KKT koşul toleransı (TOL).
     * |yᵢ·f(xᵢ) - 1| < TOL ise KKT sağlandı sayılır.
     * Çok küçük → yavaş yakınsama; çok büyük → yetersiz çözüm.
     */
    private final double tolerance;

    /**
     * Alpha güncelleme eşiği (EPS).
     * |αⱼ_new - αⱼ_old| < EPS ise bu güncelleme atlanır.
     * Sayısal gürültüyü bastırır.
     */
    private final double epsilon;

    /** Kullanılacak kernel fonksiyonu (Strategy Pattern) */
    private final KernelFunction kernel;

    // ─── Constructor ───────────────────────────────────────────────────────

    /**
     * Varsayılan hiperparametreler ile SVMTrainer oluşturur.
     *
     * @param kernel kernel fonksiyonu (örn. LinearKernel.INSTANCE)
     */
    public SVMTrainer(KernelFunction kernel) {
        this(kernel, 1e10, 500, 1e-4, 1e-5);
    }

    /**
     * Tam kontrollü constructor — tüm hiperparametreler ayarlanabilir.
     *
     * @param kernel    kernel fonksiyonu
     * @param C         düzenleme parametresi (> 0)
     * @param maxIter   maksimum iterasyon sayısı (> 0)
     * @param tolerance KKT toleransı (> 0)
     * @param epsilon   alpha eşiği (> 0)
     */
    public SVMTrainer(KernelFunction kernel, double C,
                      int maxIter, double tolerance, double epsilon) {
        if (C <= 0)          throw new SVMException("C pozitif olmalıdır: " + C);
        if (maxIter <= 0)    throw new SVMException("maxIter pozitif olmalıdır: " + maxIter);
        if (tolerance <= 0)  throw new SVMException("tolerance pozitif olmalıdır: " + tolerance);
        if (epsilon <= 0)    throw new SVMException("epsilon pozitif olmalıdır: " + epsilon);
        if (kernel == null)  throw new SVMException("kernel null olamaz");

        this.kernel    = kernel;
        this.C         = C;
        this.maxIter   = maxIter;
        this.tolerance = tolerance;
        this.epsilon   = epsilon;
    }

    // ─── Ana eğitim metodu ─────────────────────────────────────────────────

    /**
     * SMO algoritmasını çalıştırır ve SVMResult döndürür.
     *
     * @param data eğitim veri seti (en az 2 sınıf, en az 2 nokta gerekli)
     * @return eğitilmiş model (support vectorlar, ağırlıklar, bias)
     * @throws SVMException geçersiz veri veya ayrılamayan durumda
     *
     * Zaman: O(n² × maxIter)    Alan: O(n²) — kernel cache dominant
     */
    public SVMResult train(DataPoint[] data) {
        validateInput(data);

        final int n = data.length;

        // ── Veri yapılarını başlat ─────────────────────────────────────────
        double[] alpha      = new double[n];   // Lagrange çarpanları [0, C]
        double[] errorCache = new double[n];   // Eᵢ = f(xᵢ) − yᵢ önbelleği
        double   bias       = 0.0;             // b (bias / intercept)

        // ── Kernel matrisini önbellekte sakla: O(n²) ──────────────────────
        KernelCache kernelCache = new KernelCache(data, kernel);

        // ── Hata önbelleğini başlat ────────────────────────────────────────
        // f(xᵢ) = Σ αⱼyⱼK(j,i) + b = 0 + 0 = 0 başlangıçta
        // Eᵢ = f(xᵢ) − yᵢ = 0 − yᵢ = −yᵢ
        for (int i = 0; i < n; i++) {
            errorCache[i] = -data[i].getLabel();
        }

        // ── SMO Ana Döngüsü ────────────────────────────────────────────────
        int convergedAt = maxIter;

        for (int iter = 0; iter < maxIter; iter++) {
            int changedPairs = 0;

            for (int i = 0; i < n; i++) {
                double Ei = errorCache[i];
                double yi = data[i].getLabel();

                // ── KKT koşulu ihlali var mı? ──────────────────────────────
                // αᵢ < C  ve  yᵢEᵢ < -tol  (margin ihlali, sınır negatif tarafta)
                // αᵢ > 0  ve  yᵢEᵢ > +tol  (margin ihlali, sınır pozitif tarafta)
                boolean violated = (yi * Ei < -tolerance && alpha[i] < C)
                                || (yi * Ei >  tolerance && alpha[i] > 0);
                if (!violated) continue;

                // ── İkinci alpha seç: max |Ej − Ei| heuristic ─────────────
                // Açıklama: En büyük adım = en hızlı yakınsama
                int j = selectSecondAlpha(i, Ei, n, errorCache);

                double Ej = errorCache[j];
                double yj = data[j].getLabel();

                // Eski değerleri sakla (güncelleme başarısızsa geri dönmek için)
                double alphaI_old = alpha[i];
                double alphaJ_old = alpha[j];

                // ── Alpha_j sınırları: L ≤ αⱼ_new ≤ H ────────────────────
                //
                // Kısıt: αᵢyᵢ + αⱼyⱼ = sabit (s)
                //
                // yᵢ = yⱼ (aynı sınıf): αᵢ + αⱼ = sabit
                //   L = max(0, αⱼ + αᵢ − C)
                //   H = min(C, αⱼ + αᵢ)
                //
                // yᵢ ≠ yⱼ (farklı sınıf): −αᵢ + αⱼ = sabit
                //   L = max(0, αⱼ − αᵢ)
                //   H = min(C, C + αⱼ − αᵢ)
                double L, H;
                if (yi == yj) {
                    L = Math.max(0.0, alphaJ_old + alphaI_old - C);
                    H = Math.min(C,   alphaJ_old + alphaI_old);
                } else {
                    L = Math.max(0.0, alphaJ_old - alphaI_old);
                    H = Math.min(C,   C + alphaJ_old - alphaI_old);
                }
                if (L >= H - epsilon) continue;   // Geçerli aralık yok

                // ── eta = 2K(i,j) − K(i,i) − K(j,j) ─────────────────────
                // Dual'ın ikinci türevi: −eta > 0 olmalı (konveks minimize)
                // eta negatif değilse bu çift geçersiz
                double kii = kernelCache.get(i, i);
                double kjj = kernelCache.get(j, j);
                double kij = kernelCache.get(i, j);
                double eta = 2.0 * kij - kii - kjj;
                if (eta >= 0.0) continue;

                // ── Alpha_j'yi güncelle ────────────────────────────────────
                // αⱼ_new_unclipped = αⱼ − yⱼ(Eᵢ − Eⱼ) / eta
                double alphaJ_new = alphaJ_old - yj * (Ei - Ej) / eta;
                // Kırp: [L, H] aralığına sıkıştır
                alphaJ_new = clamp(alphaJ_new, L, H);

                // Yeterli değişim yoksa atla (sayısal gürültüden kaçın)
                if (Math.abs(alphaJ_new - alphaJ_old) < epsilon) continue;

                // ── Alpha_i'yi güncelle ────────────────────────────────────
                // αᵢ_new = αᵢ + yᵢyⱼ(αⱼ_old − αⱼ_new)
                double alphaI_new = alphaI_old + yi * yj * (alphaJ_old - alphaJ_new);
                alphaI_new = clamp(alphaI_new, 0.0, C);   // sayısal taşmaya karşı

                alpha[i] = alphaI_new;
                alpha[j] = alphaJ_new;

                // ── Bias (b) güncelle ──────────────────────────────────────
                // b1: αᵢ support vector olsaydı doğru b
                // b2: αⱼ support vector olsaydı doğru b
                // Her ikisi de SV değilse ortalama alınır
                double deltaI = alphaI_new - alphaI_old;
                double deltaJ = alphaJ_new - alphaJ_old;

                double b1 = bias - Ei
                          - yi * deltaI * kii
                          - yj * deltaJ * kij;

                double b2 = bias - Ej
                          - yi * deltaI * kij
                          - yj * deltaJ * kjj;

                if (alphaI_new > 0.0 && alphaI_new < C)      bias = b1;
                else if (alphaJ_new > 0.0 && alphaJ_new < C) bias = b2;
                else                                           bias = (b1 + b2) * 0.5;

                // ── Hata önbelleğini güncelle: O(n) ───────────────────────
                // Eₖ = f(xₖ) − yₖ = (Eₖ_old + Δbias + Δαᵢ·yᵢ·K(i,k) + Δαⱼ·yⱼ·K(j,k))
                // bias değişimi + iki alpha değişiminin katkısı
                double biasDelta = bias - (b1 + b2) * 0.5 + (b1 + b2) * 0.5 - (bias - b1);
                // Daha sade: tüm hataları yeniden hesapla
                double newBias = bias;
                for (int k = 0; k < n; k++) {
                    errorCache[k] += yi * deltaI * kernelCache.get(i, k)
                                   + yj * deltaJ * kernelCache.get(j, k)
                                   + (newBias - (bias - (b1 - newBias)));
                }
                // bias artık sabit, hata güncellendi; offset'i düzelt
                // Yukarıdaki döngü bias farkını iki kez ekledi, düzelt:
                recomputeErrors(n, data, alpha, bias, kernelCache, errorCache);

                changedPairs++;
            } // iç döngü sonu

            // Hiç güncelleme olmazsa: KKT koşulları sağlanmış → yakınsadık
            if (changedPairs == 0) {
                convergedAt = iter + 1;
                break;
            }
        } // dış döngü sonu

        // ── Sonuç nesnesini oluştur ────────────────────────────────────────
        SVMResult result = buildResult(data, alpha, bias, convergedAt);

        // ── Bellek temizliği ──────────────────────────────────────────────
        // KernelCache'i GC'ye bırak (n²'lik matris artık gerekmiyor)
        kernelCache.clear();
        // alpha ve errorCache yerel değişken → metod bitince otomatik serbest

        return result;
    }

    // ─── Yardımcı metodlar ─────────────────────────────────────────────────

    /**
     * Tüm hata değerlerini sıfırdan yeniden hesaplar.
     * Eᵢ = f(xᵢ) − yᵢ = (Σⱼ αⱼyⱼK(j,i) + b) − yᵢ
     *
     * Zaman: O(n²) — yalnızca gerektiğinde çağrılır
     * Tercih: incremental update yerine tam yeniden hesap → sayısal kararlılık
     */
    private void recomputeErrors(int n, DataPoint[] data,
                                  double[] alpha, double bias,
                                  KernelCache cache, double[] errorCache) {
        for (int i = 0; i < n; i++) {
            double fi = bias;
            for (int j = 0; j < n; j++) {
                fi += alpha[j] * data[j].getLabel() * cache.get(j, i);
            }
            errorCache[i] = fi - data[i].getLabel();
        }
    }

    /**
     * İkinci alpha seçimi — maximum |Eⱼ − Eᵢ| heuristic.
     *
     * NEDEN BU SEÇİM?
     *   αⱼ güncellemesi: Δαⱼ ∝ |Eᵢ − Eⱼ| / |eta|
     *   |Eᵢ − Eⱼ| maksimum → en büyük güncelleme → en hızlı yakınsama
     *
     * Zaman: O(n)
     *
     * @param i          birinci alpha indeksi
     * @param Ei         birinci hatası
     * @param n          toplam nokta sayısı
     * @param errorCache hata önbelleği
     * @return seçilen j indeksi
     */
    private int selectSecondAlpha(int i, double Ei,
                                   int n, double[] errorCache) {
        int    jBest   = -1;
        double maxDiff = -1.0;

        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            double diff = Math.abs(Ei - errorCache[k]);
            if (diff > maxDiff) {
                maxDiff = diff;
                jBest   = k;
            }
        }

        // Fallback: tüm hatalar eşitse sonraki indeksi seç
        return (jBest == -1) ? ((i + 1) % n) : jBest;
    }

    /**
     * SVMResult nesnesini oluşturur.
     * w = Σ αᵢyᵢxᵢ,  support vectorlar = {i | αᵢ > epsilon}
     *
     * Zaman: O(n)
     */
    private SVMResult buildResult(DataPoint[] data, double[] alpha,
                                   double bias, int convergedAt) {
        double w1 = 0.0, w2 = 0.0;
        List<DataPoint> svList = new ArrayList<>();

        for (int i = 0; i < data.length; i++) {
            if (alpha[i] > epsilon) {
                w1 += alpha[i] * data[i].getLabel() * data[i].getX1();
                w2 += alpha[i] * data[i].getLabel() * data[i].getX2();
                svList.add(data[i]);
            }
        }

        return new SVMResult(w1, w2, bias, alpha, svList, convergedAt);
    }

    /**
     * Girdi doğrulaması — eğitimden önce çağrılır.
     * @throws SVMException geçersiz veri için
     */
    private void validateInput(DataPoint[] data) {
        if (data == null || data.length < 2) {
            throw new SVMException(
                "En az 2 veri noktası gereklidir. Verilen: " +
                (data == null ? "null" : data.length)
            );
        }
        int positiveCount = 0, negativeCount = 0;
        for (DataPoint dp : data) {
            if (dp == null) throw new SVMException("Veri noktası null olamaz");
            if (dp.getLabel() == 1)  positiveCount++;
            if (dp.getLabel() == -1) negativeCount++;
        }
        if (positiveCount == 0) throw new SVMException("Sınıf A (+1) noktası bulunamadı");
        if (negativeCount == 0) throw new SVMException("Sınıf B (-1) noktası bulunamadı");
    }

    /**
     * Bir değeri [min, max] aralığına kırpar.
     * Math.min(Math.max(...)) yerine okunabilirlik için ayrı metod.
     * Zaman: O(1)
     */
    private double clamp(double value, double min, double max) {
        return Math.min(max, Math.max(min, value));
    }
}
