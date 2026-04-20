package svm.app;

import svm.model.DataPoint;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Application (Uygulama Katmanı)
 *  SINIF  : DatasetFactory — test veri setlerini üreten fabrika
 * ═══════════════════════════════════════════════════════════════════
 *
 *  TASARIM DESENİ: Factory Method
 *    Her statik metod farklı bir veri seti döndürür.
 *    Main sınıfı hangi verinin nasıl üretildiğini bilmek zorunda değil.
 *
 *  OOP PRENSİPLERİ:
 *    • Factory Pattern: nesne üretimini merkezi bir yerden yönetir
 *    • Single Responsibility: yalnızca veri üretir
 *    • Open/Closed: yeni veri seti → yeni static metod, var olan değişmez
 */
public final class DatasetFactory {

    /** Utility class — instance oluşturulamaz */
    private DatasetFactory() {}

    public static DataPoint[] createObstacleDataset() {
        return new DataPoint[] {
            // ── Sınıf A Engelleri (+1) — sol-alt bölge ──────────────────
            new DataPoint(1.0, 2.0,  1),
            new DataPoint(2.0, 3.0,  1),
            new DataPoint(1.5, 1.5,  1),
            new DataPoint(0.5, 3.0,  1),
            new DataPoint(2.5, 2.0,  1),
            new DataPoint(1.0, 4.0,  1),
            new DataPoint(3.0, 3.5,  1),
            new DataPoint(0.5, 1.0,  1),

            // ── Sınıf B Engelleri (-1) — sağ-üst bölge ─────────────────
            new DataPoint(6.0, 5.0, -1),
            new DataPoint(7.0, 4.0, -1),
            new DataPoint(5.5, 6.0, -1),
            new DataPoint(8.0, 5.0, -1),
            new DataPoint(6.5, 7.0, -1),
            new DataPoint(7.5, 6.0, -1),
            new DataPoint(5.0, 4.5, -1),
            new DataPoint(8.5, 4.5, -1),
        };
    }

    /**
     * Klasik XOR benzeri zor veri seti — lineer ayrılamaz.
     * Soft-margin veya kernel-SVM gerektirir.
     * (Demo amaçlı — bu ödevde kullanılmaz)
     */
    public static DataPoint[] createHardDataset() {
        return new DataPoint[] {
            new DataPoint(1.0, 1.0,  1),
            new DataPoint(3.0, 3.0,  1),
            new DataPoint(1.0, 3.0, -1),
            new DataPoint(3.0, 1.0, -1),
        };
    }
}
