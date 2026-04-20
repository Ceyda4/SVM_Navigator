package svm.kernel;

import svm.model.DataPoint;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Kernel (Hesaplama Katmanı)
 *  SINIF  : LinearKernel — lineer kernel K(a,b) = a·b
 * ═══════════════════════════════════════════════════════════════════
 *
 *  MATEMATİK:
 *    K(xᵢ, xⱼ) = xᵢ₁·xⱼ₁ + xᵢ₂·xⱼ₂   (nokta çarpımı / iç çarpım)
 *
 *    Bu kernel, orijinal uzayda lineer karar sınırı üretir.
 *    Lineer ayrılabilir veriler için en verimli seçimdir.
 *
 *  ZAMAN KARMAŞIKLİĞİ: O(d) = O(2) ≡ O(1) — 2D veri için sabit
 *  ALAN KARMAŞIKLİĞİ:  O(1) — ek bellek kullanılmaz
 *
 *  OOP PRENSİPLERİ:
 *    • Stateless: alan yok → thread-safe, paylaşılabilir singleton
 *    • Strategy: KernelFunction arayüzünü uygular
 */
public final class LinearKernel implements KernelFunction {

    /**
     * Singleton örneği — nesne üretimi gereksiz, durum yok.
     * BELLEK TASARRUFU: her SMO iterasyonunda yeni nesne yaratılmaz.
     */
    public static final LinearKernel INSTANCE = new LinearKernel();

    /** private constructor — yalnızca INSTANCE üzerinden erişim */
    private LinearKernel() {}

    /**
     * K(a, b) = a₁·b₁ + a₂·b₂
     *
     * Zaman: O(1) — sabit sayıda çarpma ve toplama
     */
    @Override
    public double compute(DataPoint a, DataPoint b) {
        return a.getX1() * b.getX1()
             + a.getX2() * b.getX2();
    }

    @Override
    public String toString() {
        return "LinearKernel{K(a,b) = a·b}";
    }
}
