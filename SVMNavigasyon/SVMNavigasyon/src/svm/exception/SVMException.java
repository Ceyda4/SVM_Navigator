package svm.exception;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Exception (Hata Yönetimi Katmanı)
 *  SINIF  : SVMException — domain-specific unchecked exception
 * ═══════════════════════════════════════════════════════════════════
 *
 *  RuntimeException'dan türetilir → checked exception zorunluluğu yok;
 *  ancak tüm fırlatma noktaları Javadoc ile belgelenmiştir.
 *
 *  OOP PRENSİPLERİ:
 *    • Single Responsibility: yalnızca SVM hata durumlarını temsil eder
 *    • Inheritance: RuntimeException zinciri korunur
 *    • Encapsulation: cause chain tutulur (Throwable cause)
 */
public class SVMException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /** @param message hata açıklaması */
    public SVMException(String message) {
        super(message);
    }

    /** @param message hata açıklaması, @param cause köken istisna */
    public SVMException(String message, Throwable cause) {
        super(message, cause);
    }
}
