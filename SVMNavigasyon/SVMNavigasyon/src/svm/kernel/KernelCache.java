package svm.kernel;

import svm.model.DataPoint;

/**
 * ═══════════════════════════════════════════════════════════════════
 *  KATMAN : Kernel (Hesaplama Katmanı)
 *  SINIF  : KernelCache — kernel matrisini önbelleğe alır
 * ═══════════════════════════════════════════════════════════════════
 *
 *  NEDEN GEREKLİ?
 *    SMO algoritmasında K(i,j) değeri her iterasyonda tekrar tekrar
 *    hesaplanır. n=100 nokta, 300 iter → 100²×300 = 3.000.000 hesap.
 *    Önbellek ile: n² = 10.000 hesap (bir kere), sonra O(1) lookup.
 *
 *  BELLEK ANALİZİ:
 *    n×n double matris = n² × 8 byte
 *    n=100  → 80 KB    (kabul edilebilir)
 *    n=1000 → 8 MB     (dikkatli kullanım)
 *    n=5000 → 200 MB   (büyük veri için chunk stratejisi gerekir)
 *
 *  ZAMAN KARMAŞIKLİĞİ:
 *    Ön-hesaplama (precompute): O(n²)     — bir kere
 *    Lookup sonrası:            O(1)      — dizi erişimi
 *    Toplam SMO ile:            O(n²)     — ön-hesap dominating
 *
 *  ALAN KARMAŞIKLİĞİ:  O(n²)
 *    Simetrik matris olduğundan yarısı yeterli olurdu (n²/2),
 *    ancak kod basitliği için tam matris kullanılır.
 *
 *  OOP PRENSİPLERİ:
 *    • Encapsulation: matris private, yalnızca get() ile erişim
 *    • Single Responsibility: yalnızca önbellekleme yapar
 *    • Composition: KernelFunction'ı içerir (is-a değil has-a)
 *
 *  BELLEK GÜVENLİĞİ:
 *    • cache dizisi yalnızca constructor'da tahsis edilir
 *    • Dışarıya referans verilmez → sızıntı riski sıfır
 *    • clear() metodu çöp toplayıcıya ipucu verir (null atama)
 */
public final class KernelCache {

    /** Önbellek matrisi: cache[i][j] = K(xᵢ, xⱼ) */
    private double[][] cache;

    /** Kaç noktanın kernel değeri önbelleğe alındı */
    private final int size;

    // ─── Constructor ───────────────────────────────────────────────────────

    /**
     * Tüm kernel değerlerini önceden hesaplar.
     * Zaman: O(n²)    Alan: O(n²)
     *
     * @param data   eğitim noktaları
     * @param kernel kullanılacak kernel fonksiyonu
     */
    public KernelCache(DataPoint[] data, KernelFunction kernel) {
        this.size  = data.length;
        this.cache = new double[size][size];

        // Tam matris hesabı — simetrik: K(i,j)=K(j,i), ancak 2x hesap
        // kod karmaşıklığını azaltmak için kabul edilir
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {           // j = i'den başla (simetri)
                double val = kernel.compute(data[i], data[j]);
                cache[i][j] = val;
                cache[j][i] = val;                      // simetri
            }
        }
    }

    // ─── Erişim ────────────────────────────────────────────────────────────

    /**
     * K(i, j) değerini O(1) zamanda döndürür.
     *
     * @param i birinci indeks [0, size)
     * @param j ikinci indeks  [0, size)
     * @return önbelleğe alınmış kernel değeri
     * @throws svm.exception.SVMException indeks sınır dışıysa
     */
    public double get(int i, int j) {
        if (i < 0 || i >= size || j < 0 || j >= size) {
            throw new svm.exception.SVMException(
                "KernelCache indeks sınır dışı: i=" + i + " j=" + j +
                " size=" + size
            );
        }
        return cache[i][j];
    }

    /** @return matris boyutu (nokta sayısı) */
    public int getSize() { return size; }

    /**
     * Belleği serbest bırakır — büyük ön-belleklerde GC'ye ipucu.
     * Çağrıldıktan sonra bu nesne kullanılamaz.
     *
     * BELLEK YÖNETİMİ NOTU:
     *   Java'da manuel free() yoktur; ancak büyük dizilere null atamak
     *   GC'nin bir sonraki döngüde belleği geri almasını sağlar.
     *   Bu, C/C++'daki delete[] ile eşdeğer semantiğe sahiptir.
     */
    public void clear() {
        this.cache = null;
    }
}
