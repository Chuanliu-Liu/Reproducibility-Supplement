#include <fenv.h>
#include <float.h>
#include <stddef.h>
#pragma STDC FENV_ACCESS ON
/* Build: gcc -O2 -std=c11 -shared -fPIC -frounding-math
 * -ffp-contract=off -fno-fast-math alm_directed.c -lm. No fused operations.
 * Each product and addition is rounded in the specified direction. Exact zero
 * terms are skipped. Gradual underflow is required and tested by the caller.
 * Restore the caller's rounding mode on every return; one numerical thread.
 */
int alm_round(void) { return fegetround(); }
int alm_mantissa(int extended) { return extended ? LDBL_MANT_DIG : DBL_MANT_DIG; }
#define KERNEL(NAME, TYPE) \
int NAME(TYPE *out, const TYPE *in, size_t rows, size_t cols, \
         size_t old_cols, size_t dr, size_t dc, TYPE p, int upper) { \
    int saved=fegetround(); \
    if (fesetround(upper ? FE_UPWARD : FE_DOWNWARD)) return -1; \
    for (size_t r=0; r<rows; r++) \
        for (size_t c=0; c<cols; c++) { \
            TYPE a=in[(r+dr)*old_cols+c+dc]; \
            if (p != 0 && a != 0) { \
                volatile TYPE product=a*p; \
                out[r*cols+c]=out[r*cols+c]+product; \
            } \
        } \
    return fesetround(saved); \
}
KERNEL(alm_shift64, double)
KERNEL(alm_shiftld, long double)
