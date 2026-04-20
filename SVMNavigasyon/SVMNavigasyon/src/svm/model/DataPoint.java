package svm.model;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Model (Veri Katmanı)
 *  SINIF  : DataPoint — 2D koordinat + sınıf etiketi
 * ═══════════════════════════════════════════════════════════════════
 *
 *  OOP PRENSİPLERİ:
 *    • Immutable (değiştirilemez): tüm alanlar final → bellek güvenliği
 *    • Encapsulation: alanlar private, erişim yalnızca getter ile
 *    • Value Object: equals/hashCode koordinat+etiket üzerinden
 *
 *  BELLEK YÖNETİMİ:
 *    • final alanlar → JIT tarafından register'a alınabilir
 *    • Nesne oluşturulunca hiçbir alan değişmez → thread-safe by design
 *    • Primitive double/int → wrapper Boxing overhead yok
 */
public final class DataPoint {

    /** x₁ koordinatı (yatay eksen) */
    private final double x1;

    /** x₂ koordinatı (dikey eksen) */
    private final double x2;

    /**
     * Sınıf etiketi.
     *   +1 → Sınıf A (birinci engel grubu)
     *   -1 → Sınıf B (ikinci engel grubu)
     */
    private final int label;

    // ─── Constructor ───────────────────────────────────────────────────────

    /**
     * @param x1    Yatay koordinat
     * @param x2    Dikey koordinat
     * @param label +1 veya -1 (diğer değer IllegalArgumentException fırlatır)
     * @throws svm.exception.SVMException geçersiz etiket için
     */
    public DataPoint(double x1, double x2, int label) {
        if (label != 1 && label != -1) {
            throw new svm.exception.SVMException(
                "Geçersiz etiket: " + label + ". Yalnızca +1 veya -1 kabul edilir."
            );
        }
        if (Double.isNaN(x1) || Double.isNaN(x2) ||
            Double.isInfinite(x1) || Double.isInfinite(x2)) {
            throw new svm.exception.SVMException(
                "Koordinat NaN veya Infinite olamaz: (" + x1 + ", " + x2 + ")"
            );
        }
        this.x1    = x1;
        this.x2    = x2;
        this.label = label;
    }

    // ─── Getters ───────────────────────────────────────────────────────────

    /** @return x₁ koordinatı */
    public double getX1() { return x1; }

    /** @return x₂ koordinatı */
    public double getX2() { return x2; }

    /** @return sınıf etiketi: +1 veya -1 */
    public int getLabel() { return label; }

    /** @return insan okunabilir sınıf adı */
    public String getClassName() { return label == 1 ? "A" : "B"; }

    // ─── Object overrides ──────────────────────────────────────────────────

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DataPoint)) return false;
        DataPoint dp = (DataPoint) o;
        return Double.compare(dp.x1, x1) == 0
            && Double.compare(dp.x2, x2) == 0
            && dp.label == label;
    }

    @Override
    public int hashCode() {
        int result = Double.hashCode(x1);
        result = 31 * result + Double.hashCode(x2);
        result = 31 * result + label;
        return result;
    }

    @Override
    public String toString() {
        return String.format("DataPoint{x1=%.4f, x2=%.4f, sinif=%s}",
                             x1, x2, getClassName());
    }
}
