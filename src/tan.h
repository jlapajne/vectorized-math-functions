#pragma once

#include "CommonUtils.h"

#include <hwy/highway.h>

#include <array>
#include <immintrin.h>

#include <Polynomials.h>

namespace trigon {
namespace {

// clang-format off
template <typename T>
    requires std::is_floating_point_v<T>
inline constexpr static std::array<T, 12> trigonTanCoeffs = 
    {34459425, 4729725, 135135, 990};
// clang-format on
} // namespace

template <typename T, Approximation A>
struct TanPadeSeriesCoeffs {};

template <typename T>
inline Vec<T> tan(Vec<T> const &x) {}

template <typename T>
inline Vec<T> tan_ver2(Vec<T> const &x) {
    Vec<T> x2 = hw::Mul(x, x);
    Vec<T> numerator = hw::Sub(x2, hw::Set(d<T>, T(55)));
    Vec<T> coeff1 = hw::Set(d<T>, T(654729075));
    numerator = hw::MulAdd(x, hw::Mul(x, numerator), hw::Set(d<T>, T(25740)));
    numerator = hw::MulSub(x, hw::Mul(x, numerator), hw::Set(d<T>, T(2837835)));
    numerator = hw::MulAdd(x, hw::Mul(x, numerator), hw::Set(d<T>, T(91891800)));
    numerator = hw::MulSub(x, hw::Mul(x, numerator), coeff1);
    numerator = hw::Mul(x, numerator);

    Vec<T> denominator = hw::Sub(x2, _mm_set_ps1(-1485));
    denominator = hw::MulAdd(x, hw::Mul(x, denominator), hw::Set(d<T>, T(315315)));
    denominator = hw::MulSub(x, hw::Mul(x, denominator), hw::Set(d<T>, T(18918900)));
    denominator = hw::MulAdd(x, hw::Mul(x, denominator), hw::Set(d<T>, T(310134825)));
    denominator = hw::MulSub(x, hw::Mul(x, denominator), coeff1);

    return hw::Div(numerator, denominator);
}

template <typename T>
inline Vec<T> tan_ver3(Vec<T> const &x) {
    // version 4 S(4,a)
    // this function should calculate tan to 1e-8 precision
    // stevec : 34459425 * a - 4729725 * a ^ 3 + 135135 * a ^ 5 -
    // 990 a ^ 7 + a ^ 9;
    Vec<T> x2 = hw::Mul(x, x);
    Vec<T> coeff1 = hw::Set(d<T>, T(34459425));
    Vec<T> numerator = hw::Sub(x2, T(-990));

    numerator = hw::MulAdd(x, hw::Mul(x, numerator), hw::Set(d<T>, 135135.f));
    numerator = hw::MulSub(x, hw::Mul(x, numerator), hw::Set(d<T>, 4729725.f));
    numerator = hw::MulAdd(x, hw::Mul(x, numerator), coeff1);
    numerator = hw::Mul(x, numerator);

    // imenovalec: 34459425 - 16216200*a^2 + 945945 a^4 - 13860*a^6
    // + 45*a^8
    Vec<T> denominator = hw::Add(x2, _mm_set_ps1(45));
    denominator = hw::MulSub(x, hw::Mul(x, denominator), hw::Set(d<T>, T(13860)));
    denominator = hw::MulAdd(x, hw::Mul(x, denominator), hw::Set(d<T>, T(945945)));
    denominator = hw::MulSub(x, hw::Mul(x, denominator), hw::Set(d<T>, T(16216200)));
    denominator = hw::MulAdd(x, hw::Mul(x, denominator), coeff1);
    return hw::Div(numerator, denominator);
}

template <typename T>
inline Vec<T> arctan(Vec<T> const &x0) {
    // this comparison sets 0xffff if true - which is not 1.0
    Vec<T> cmp1 = hw::Gt(Zero(d<T>), x0);
    // now we compare to 1.0 to get desired 0.0 and 1.0 numbers
    cmp1 = hw::And(cmp1, hw::Set(d<T>, T(-1)));
    Vec<T> cmp1x2 = hw::Mul(hw::Set(d<T>, T(2)), cmp1);

    // cmp1 =_mm_xor_ps(v, _mm_set1_ps(-0.0)); convert 1 to -1
    // now calculate absolute value
    Vec<T> x = hw::MulAdd(cmp1x2, x0, x0);

    Vec<T> cmp2 = hw::Gt(x, _mm_set_ps1(1.f));
    // cmp2 = _mm_and_ps(cmp2, _mm_set_ps1(1.0));

    Vec<T> overx = hw::Div(hw::Set(d<T>, T(1)), x);
    x = _mm_blendv_ps(x, overx, cmp2);

    Vec<T> numerator = hw::MulAdd(hw::Mul(x, hw::Set(14928225)), x, hw::Set(341536195));
    numerator = hw::MulAdd(hw::Mul(x, numerator), x, hw::Set(2146898754));
    numerator = hw::MulAdd(hw::Mul(x, numerator), x, hw::Set(5429886462));
    numerator = hw::MulAdd(hw::Mul(x, numerator), x, hw::Set(5941060125));
    numerator = hw::MulAdd(hw::Mul(x, numerator), x, hw::Set(2342475135.f));
    numerator = hw::Mul(numerator, x);

    // {2342475135, 0, 6721885170, 0, 7202019825, 0, 3537834300, 0,
    // 780404625, 0, 62432370, 0, 800415}
    Vec<T> denominator =
        hw::MulAdd(x, hw::Mul(x, hw::Set(d<T>, T(800415))), hw::Set(d<T>, T(62432370)));
    denominator = hw::MulAdd(x, hw::Mul(denominator, x), hw::Set(T(780404625)));
    denominator = hw::MulAdd(x, hw::Mul(denominator, x), hw::Set(T(353783430)));
    denominator = hw::MulAdd(x, hw::Mul(denominator, x), hw::Set(T(720201982)));
    denominator = hw::MulAdd(x, hw::Mul(denominator, x), hw::Set(T(672188517)));
    denominator = hw::MulAdd(x, hw::Mul(denominator, x), hw::Set(T(234247513)));

    __m128 result = _mm_div_ps(numerator, denominator);
    __m128 pi2 = _mm_set_ps1(PI_2<float>);
    __m128 x_pi = _mm_sub_ps(pi2, result);
    result = _mm_blendv_ps(result, x_pi, cmp2);
    cmp1 = _mm_fmadd_ps(cmp1, _mm_set_ps1(2.f), _mm_set_ps1(1.f));

    result = _mm_mul_ps(result, cmp1);
    return result;
}

/** @brief The function calculates arcsin of input vector. Input
 * vector should contain elements in range [-1.0, 1.0].
 * @details This is sse float implementation.
 * @param t input sse vector of values for which arcsin is
 * calculated
 */
inline __attribute__((always_inline)) __m128 arcsin(const __m128 &t) {

    __m128 twos = _mm_set1_ps(2.0);
    __m128 ones = _mm_set1_ps(1.0);
    __m128 t2 = _mm_mul_ps(t, t);

    __m128 sqrt = _mm_set1_ps(0.70710678118654752440);
    __m128 mask1 = _mm_cmpgt_ps(t, sqrt);
    __m128 mask2 = _mm_cmplt_ps(t, _mm_xor_ps(sqrt, SignMask<T>));
    __m128 combined_mask = _mm_or_ps(mask1, mask2);
    __m128 sqrtt = _mm_sqrt_ps(_mm_sub_ps(ones, t2));

    __m128 t_mod = _mm_blendv_ps(t, sqrtt, combined_mask);
    t2 = _mm_mul_ps(t_mod, t_mod);

    __m128 dn = ones;
    __m128 cn = twos;
    __m128 sum = t_mod;
    __m128 next_t = _mm_mul_ps(t2, t_mod);
    __m128 counter2 = twos;
    __m128 i_vec = _mm_set1_ps(3.0);

    // determine the number of iterations needed
    float max_element = _mm_horizontal_max_ps(_mm_andnot_ps(SignMask<T>, t_mod));

    int max_iter = std::ceil(-5 / std::log10(std::fabs(max_element)));

    for (int i = 3; i <= max_iter; i += 2) {

        __m128 factor = _mm_div_ps(dn, _mm_mul_ps(cn, i_vec));
        sum = _mm_fmadd_ps(factor, next_t, sum);

        next_t = _mm_mul_ps(next_t, t2);
        dn = _mm_mul_ps(dn, i_vec);

        counter2 = _mm_add_ps(counter2, twos);
        cn = _mm_mul_ps(cn, counter2);

        i_vec = _mm_add_ps(i_vec, twos);
    }
    static const __m128 pi2 = _mm_set1_ps(PI_2<float>); // pi/2
    __m128 sol1 = _mm_sub_ps(pi2, sum);
    __m128 sol2 = _mm_add_ps(_mm_xor_ps(pi2, SignMask<T>), sum);

    return _mm_blendv_ps(_mm_blendv_ps(sum, sol1, mask1), sol2, mask2);
}

/** @brief The function calculates arcsin of input vector. Input
 * vector should contain elements in range [-1.0, 1.0].
 * @details This is sse float implementation.
 * @param t input sse vector of values for which arcsin is
 * calculated
 */
inline __attribute__((always_inline)) __m128 arccos(const __m128 &t) {
    return _mm_sub_ps(_mm_set1_ps(1.5707963267948966192), arcsin(t));
}

#ifdef __AVX2__
/** @brief The function calculates arcsin of input vector. Input
 * vector should contain elements in range [-1.0, 1.0].
 * @details This is avx double implementation.
 * @param t input avx vector of values for which double precision
 * arcsin is calculated.
 */
inline __attribute__((always_inline)) __m256d arcsin(const __m256d &t) {

    __m256d twos = _mm256_set1_pd(2.0);
    __m256d ones = _mm256_set1_pd(1.0);
    __m256d t2 = _mm256_mul_pd(t, t);

    __m256d sqrt = _mm256_set1_pd(0.70710678118654752440);
    __m256d mask1 = _mm256_cmp_pd(t, sqrt, _CMP_GT_OS);
    __m256d mask2 = _mm256_cmp_pd(t, _mm256_xor_pd(sqrt, SIGNMASK256d), _CMP_LT_OS);

    __m256d combined_mask = _mm256_or_pd(mask1, mask2);
    __m256d sqrtt = _mm256_sqrt_pd(_mm256_sub_pd(ones, t2));

    __m256d t_mod = _mm256_blendv_pd(t, sqrtt, combined_mask);

    t2 = _mm256_mul_pd(t_mod, t_mod);

    __m256d dn = ones;
    __m256d cn = twos;
    __m256d sum = t_mod;
    __m256d next_t = _mm256_mul_pd(t2, t_mod);
    __m256d counter2 = twos;
    __m256d i_vec = _mm256_set1_pd(3.0);

    // determine the number of iterations needed
    double max_element = _mm256_horizontal_max_pd(_mm256_andnot_pd(SIGNMASK256d, t_mod));

    int max_iter = std::ceil(-12 / std::log10(std::fabs(max_element)));

    for (int i = 3; i <= max_iter; i += 2) {

        __m256d factor = _mm256_div_pd(dn, _mm256_mul_pd(cn, i_vec));

        sum = _mm256_fmadd_pd(factor, next_t, sum);
        next_t = _mm256_mul_pd(next_t, t2);
        dn = _mm256_mul_pd(dn, i_vec);

        counter2 = _mm256_add_pd(counter2, twos);
        cn = _mm256_mul_pd(cn, counter2);

        i_vec = _mm256_add_pd(i_vec, twos);
    }
    static const __m256d pi2 = _mm256_set1_pd(PI_2<double>); // pi/2
    __m256d sol1 = _mm256_sub_pd(pi2, sum);
    __m256d sol2 = _mm256_add_pd(_mm256_xor_pd(pi2, SIGNMASK256d), sum);

    return _mm256_blendv_pd(_mm256_blendv_pd(sum, sol1, mask1), sol2, mask2);
}
#endif

#ifdef __AVX2__
/** @brief The function calculates arcsin of input vector. Input
 * vector should contain elements in range [-1.0, 1.0].
 * @details This is avx double implementation.
 * @param t input avx vector of values for which double precision
 * arcsin is calculated.
 */
inline __attribute__((always_inline)) __m256d arccos(const __m256d &t) {
    return _mm256_sub_pd(_mm256_set1_pd(PI_2<double>), arcsin(t));
};
#endif
} // namespace trigon