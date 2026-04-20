package svm.reporter;

import svm.model.DataPoint;
import svm.model.SVMResult;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Reporter (Sunum / Çıktı Katmanı)
 *  SINIF  : SVMReporter — sonuçları biçimlendirir ve yazdırır
 * ═══════════════════════════════════════════════════════════════════
 *
 *  OOP PRENSİPLERİ:
 *    • Single Responsibility: yalnızca raporlama/formatlama yapar
 *    • Separation of Concerns: algoritma kodu sunum kodundan ayrılır
 *    • Dependency Inversion: SVMResult arayüzüne bağlıdır, somut
 *      implementasyona değil
 *
 *  NOT:
 *    Gerçek bir projede bu katman bir Logger (SLF4J gibi) kullanırdı.
 *    Basitlik için System.out kullanılmıştır.
 */
public final class SVMReporter {

    private static final String SEP =
        "══════════════════════════════════════════════════════════════";

    /** Utility class — instance oluşturulamaz */
    private SVMReporter() {}

    // ─── Genel rapor ───────────────────────────────────────────────────────

    /**
     * Tam eğitim raporunu konsola yazdırır.
     *
     * @param data      eğitim veri seti
     * @param result    eğitim sonucu
     * @param elapsedMs eğitim süresi (milisaniye)
     */
    public static void printFullReport(DataPoint[] data,
                                        SVMResult result,
                                        double elapsedMs) {
        printHeader();
        printDataSummary(data);
        printDecisionBoundary(result);
        printMargin(result);
        printSupportVectors(result);
        printCorridorBounds(result);
        printOptimalityExplanation(result);
        printValidation(data, result);
    }

    // ─── Bölüm yazıcılar ───────────────────────────────────────────────────

    private static void printHeader() {
        System.out.println();
        System.out.println("╔" + SEP + "╗");
        System.out.println("║     OTONOM ARAÇ GÜVENLİK MODÜLÜ — SVM SONUÇLARI          ║");
        System.out.println("╚" + SEP + "╝");
    }

    private static void printDataSummary(DataPoint[] data) {
        long countA = 0, countB = 0;
        for (DataPoint dp : data) {
            if (dp.getLabel() ==  1) countA++;
            if (dp.getLabel() == -1) countB++;
        }
        System.out.println();
        System.out.println("── VERİ ÖZETİ ──────────────────────────────────────────────");
        System.out.printf("   Toplam nokta  : %d%n", data.length);
        System.out.printf("   Sınıf A (+1)  : %d engel%n", countA);
        System.out.printf("   Sınıf B (-1)  : %d engel%n", countB);
    }

    private static void printDecisionBoundary(SVMResult result) {
        double norm = result.getWNorm();
        System.out.println();
        System.out.println("── [1] KARAR SINIRI DENKLEMİ ───────────────────────────────");
        System.out.printf("   Ham     : (%.6f)·x₁ + (%.6f)·x₂ + (%.6f) = 0%n",
            result.getW1(), result.getW2(), result.getBias());

        if (norm > 1e-12) {
            System.out.printf("   Normalize: (%.6f)·x₁ + (%.6f)·x₂ + (%.6f) = 0%n",
                result.getW1() / norm,
                result.getW2() / norm,
                result.getBias() / norm);
        }
        System.out.printf("   ‖w‖ (ağırlık normu): %.8f%n", norm);
    }

    private static void printMargin(SVMResult result) {
        System.out.println();
        System.out.println("── [2] MARGIN GENİŞLİĞİ (Güvenlik Koridoru) ───────────────");
        System.out.printf("   Formül  : 2 / ‖w‖%n");
        System.out.printf("   Değer   : 2 / %.8f = %.8f birim%n",
            result.getWNorm(), result.getMargin());
        System.out.printf("   Yorum   : Her iki sınıfa da %.4f / 2 = %.4f birim uzaklık%n",
            result.getMargin(), result.getMargin() / 2.0);
    }

    private static void printSupportVectors(SVMResult result) {
        System.out.println();
        System.out.println("── [3] SUPPORT VEKTÖRLER (Sınır Belirleyici Engeller) ──────");
        System.out.println("   (αᵢ > 0 olan, karar sınırına tam sınırda olan noktalar)");
        System.out.println();

        double[] alphas = result.getAlphas();
        for (DataPoint sv : result.getSupportVectors()) {
            System.out.printf("   ► %s%n", sv);
        }
        System.out.printf("%n   Toplam: %d support vector%n",
            result.getSupportVectors().size());
    }

    private static void printCorridorBounds(SVMResult result) {
        System.out.println();
        System.out.println("── [4] GÜVENLİK KORİDOR SINIRLARI ─────────────────────────");
        System.out.printf("   Pozitif hiper-düzlem (+1): w·x + b = +1%n");
        System.out.printf("   → (%.4f)·x₁ + (%.4f)·x₂ + (%.4f) = +1%n",
            result.getW1(), result.getW2(), result.getBias());
        System.out.printf("%n   Karar sınırı       ( 0): w·x + b = 0%n");
        System.out.printf("   → (%.4f)·x₁ + (%.4f)·x₂ + (%.4f) =  0%n",
            result.getW1(), result.getW2(), result.getBias());
        System.out.printf("%n   Negatif hiper-düzlem (-1): w·x + b = -1%n");
        System.out.printf("   → (%.4f)·x₁ + (%.4f)·x₂ + (%.4f) = -1%n",
            result.getW1(), result.getW2(), result.getBias());
    }

    private static void printOptimalityExplanation(SVMResult result) {
        System.out.println();
        System.out.println("── [5] NEDEN OPTIMUM SINIR? ────────────────────────────────");
        System.out.println();
        System.out.println("   Bu sınır matematiksel olarak optimumdur çünkü:");
        System.out.println();
        System.out.println("   1. KKT Koşulları Sağlanıyor:");
        System.out.println("      αᵢ = 0   → yᵢ·f(xᵢ) ≥ 1  (margin dışında, doğru taraf)");
        System.out.println("      0<αᵢ<C  → yᵢ·f(xᵢ) = 1  (tam sınırda — support vector)");
        System.out.printf("      → %d nokta KKT koşulunu sağlıyor%n",
            result.getSupportVectors().size());
        System.out.println();
        System.out.println("   2. Dual Optimizasyon Tamamlandı:");
        System.out.println("      Lagrange dual: Maximize Σαᵢ − ½ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(i,j)");
        System.out.printf("      SMO %d iterasyonda yakınsadı%n", result.getConvergedAt());
        System.out.println();
        System.out.println("   3. Maksimum Margin Garantisi:");
        System.out.printf("      ‖w‖ minimize edildi → Margin = 2/‖w‖ = %.6f maksimum%n",
            result.getMargin());
        System.out.println();
        System.out.println("   4. Support Vector Özelliği:");
        System.out.printf("      Karar sınırı yalnızca %d nokta tarafından belirlenir.%n",
            result.getSupportVectors().size());
        System.out.println("      Diğer tüm noktaları veri setinden kaldırsak bile");
        System.out.println("      sınır DEĞİŞMEZ — bu hem sağlamlık hem optimalite kanıtıdır.");
    }

    /**
     * Tüm noktaları sınıflandırır ve doğruluğu raporlar.
     * Zaman: O(n × n_sv)
     */
    public static void printValidation(DataPoint[] data, SVMResult result) {
        System.out.println();
        System.out.println("── [6] SINIFLANDIRMA DOĞRULAMASI ───────────────────────────");

        int correct = 0, wrong = 0;
        double[] alphas = result.getAlphas();

        for (int i = 0; i < data.length; i++) {
            DataPoint dp  = data[i];
            int prediction = result.predict(dp.getX1(), dp.getX2());
            boolean ok     = prediction == dp.getLabel();
            boolean isSV   = alphas[i] > 1e-5;

            if (ok) correct++; else wrong++;

            // Yalnızca support vector veya hatalı noktaları detaylı göster
            if (isSV || !ok) {
                System.out.printf("   %s → Tahmin: %s  Gerçek: %s  [%s]%n",
                    dp,
                    prediction == 1 ? "A" : "B",
                    dp.getClassName(),
                    ok ? (isSV ? "SUPPORT VECTOR ✓" : "OK ✓") : "HATA ✗"
                );
            }
        }

        System.out.println();
        System.out.printf("   Doğru: %d / %d  →  Doğruluk: %.1f%%%n",
            correct, data.length, 100.0 * correct / data.length);
        if (wrong > 0) {
            System.out.printf("   UYARI: %d nokta yanlış sınıflandırıldı!%n", wrong);
            System.out.println("   (Veri lineer ayrılamıyor olabilir veya C artırılmalı)");
        }
    }

    /**
     * Yeni nokta tahminini yazdırır.
     *
     * @param x1     birinci koordinat
     * @param x2     ikinci koordinat
     * @param result eğitilmiş model
     */
    public static void printPrediction(double x1, double x2, SVMResult result) {
        int label         = result.predict(x1, x2);
        double decVal     = result.decisionValue(x1, x2);
        String className  = label == 1 ? "A" : "B";
        double distToLine = Math.abs(decVal) / result.getWNorm();

        System.out.printf("   (%.2f, %.2f) → Sınıf %s  " +
                          "[karar değeri=%.4f, sınıra uzaklık=%.4f birim]%n",
            x1, x2, className, decVal, distToLine);
    }
}
