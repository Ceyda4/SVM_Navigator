package svm.kernel;

import svm.model.DataPoint;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Kernel (Hesaplama Katmanı)
 *  SINIF  : KernelFunction — kernel fonksiyonu arayüzü
 * ═══════════════════════════════════════════════════════════════════
 *
 *  TASARIM DESENİ: Strategy Pattern
 *    Farklı kernel fonksiyonları (Linear, RBF, Polynomial) aynı
 *    arayüzü uygular. SVMTrainer hangi kerneli kullandığını bilmez;
 *    yalnızca compute() çağırır. Bu Open/Closed prensibini sağlar:
 *    yeni kernel eklemek için mevcut kodu değiştirmeye gerek yok.
 *
 *  OOP PRENSİPLERİ:
 *    • Interface Segregation: tek sorumluluk — kernel değeri üret
 *    • Open/Closed: yeni kernel → yeni sınıf, mevcut kod dokunulmaz
 *    • Polymorphism: LinearKernel, RBFKernel aynı tipte kullanılır
 *
 *  MATEMATİK:
 *    K(xᵢ, xⱼ) — iki noktanın "benzerliği"
 *    Lineer:      K = xᵢ·xⱼ
 *    RBF/Gauss:   K = exp(-γ‖xᵢ−xⱼ‖²)
 *    Polynomial:  K = (xᵢ·xⱼ + c)^d
 */
@FunctionalInterface
public interface KernelFunction {

    /**
     * İki nokta arasındaki kernel değerini hesaplar.
     *
     * Zaman karmaşıklığı implementasyona göre değişir:
     *   Linear kernel:     O(d)   — d: boyut sayısı
     *   RBF kernel:        O(d)   — uzaklık hesabı
     *   Polynomial kernel: O(d)   — nokta çarpımı + üs
     *
     * @param a birinci nokta
     * @param b ikinci nokta
     * @return kernel değeri K(a, b)
     */
    double compute(DataPoint a, DataPoint b);
}
