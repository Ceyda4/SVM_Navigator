package svm.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Model (Veri Katmanı)
 *  SINIF  : SVMResult — eğitim sonucunu taşıyan değer nesnesi
 * ═══════════════════════════════════════════════════════════════════
 *
 *  Matematiksel içerik:
 *    Karar sınırı:  w₁·x₁ + w₂·x₂ + b = 0
 *    Margin:        2 / ‖w‖
 *    SV listesi:    αᵢ > EPS olan noktalar
 *
 *  OOP PRENSİPLERİ:
 *    • Immutable: tüm alanlar final, List defensively-copied
 *    • Single Responsibility: yalnızca sonucu taşır, hesaplama yapmaz
 *    • Encapsulation: iç liste dışarıdan değiştirilemez (unmodifiable view)
 *
 *  BELLEK YÖNETİMİ:
 *    • supportVectors listesi defensively copied → dışarıdan mutasyon yok
 *    • Collections.unmodifiableList → ekstra sarmalayıcı nesneden başka
 *      maliyet yok; clone() gerektirmez
 */
public final class SVMResult {

    /** Ağırlık vektörü: w₁ (x₁ ekseni bileşeni) */
    private final double w1;

    /** Ağırlık vektörü: w₂ (x₂ ekseni bileşeni) */
    private final double w2;

    /** Bias (eşik) terimi: b */
    private final double bias;

    /** Ağırlık vektörü normu ‖w‖ */
    private final double wNorm;

    /** Margin genişliği = 2 / ‖w‖ */
    private final double margin;

    /** Lagrange çarpanları (α dizisi, tüm eğitim noktaları için) */
    private final double[] alphas;

    /** Support vector olan noktalar (αᵢ > EPS) */
    private final List<DataPoint> supportVectors;

    /** SMO'nun kaç iterasyonda yakınsadığı */
    private final int convergedAt;

    // ─── Constructor ───────────────────────────────────────────────────────

    /**
     * Yalnızca SVMTrainer tarafından çağrılır.
     * Dışarıdan nesne üretilmesini önlemek için package-private değil public
     * bırakıldı (farklı paketten erişim gerekiyor); ancak parametre listesi
     * kasıtlı olarak karmaşık tutulmuştur — doğrudan kullanım caydırılır.
     */
    public SVMResult(double w1, double w2, double bias,
                     double[] alphas, List<DataPoint> supportVectors,
                     int convergedAt) {
        this.w1   = w1;
        this.w2   = w2;
        this.bias = bias;

        double norm = Math.sqrt(w1 * w1 + w2 * w2);
        this.wNorm  = norm;
        this.margin = (norm > 1e-12) ? 2.0 / norm : 0.0;

        // Defensive copy: dışarıdaki dizi değişirse bu nesne etkilenmez
        this.alphas = alphas.clone();

        // Defensive copy + unmodifiable view
        this.supportVectors = Collections.unmodifiableList(
            new ArrayList<>(supportVectors)
        );
        this.convergedAt = convergedAt;
    }

    // ─── Getters ───────────────────────────────────────────────────────────

    public double getW1()           { return w1; }
    public double getW2()           { return w2; }
    public double getBias()         { return bias; }
    public double getWNorm()        { return wNorm; }
    public double getMargin()       { return margin; }
    public double[] getAlphas()     { return alphas.clone(); } // defensive copy on read
    public int getConvergedAt()     { return convergedAt; }

    /** Support vector listesi — değiştirilemez görünüm (unmodifiable view) */
    public List<DataPoint> getSupportVectors() { return supportVectors; }

    /**
     * Bir koordinat için ham karar fonksiyonu değeri.
     * f(x) = w₁·x₁ + w₂·x₂ + b
     * Zaman: O(1) — doğrusal model, kernel toplamı gerekmez
     *
     * NOT: Bu değer yalnızca lineer SVM için geçerlidir.
     *      Kernel-SVM sonuçlarında bu metod kullanılmamalıdır.
     */
    public double decisionValue(double x1, double x2) {
        return w1 * x1 + w2 * x2 + bias;
    }

    /**
     * Sınıf tahmini: f(x) ≥ 0 → +1, aksi → -1
     * Zaman: O(1)
     */
    public int predict(double x1, double x2) {
        return decisionValue(x1, x2) >= 0.0 ? 1 : -1;
    }

    @Override
    public String toString() {
        return String.format(
            "SVMResult{w=(%.6f, %.6f), b=%.6f, margin=%.6f, sv_count=%d, iter=%d}",
            w1, w2, bias, margin, supportVectors.size(), convergedAt
        );
    }
}
