#pragma once

#include "CommonUtils.h"

#include <hwy/highway.h>

#include <array>
#include <immintrin.h>

#include <Polynomials.h>

namespace trigon {

// clang-format off
template <typename T>
    requires std::is_floating_point_v<T>
inline constexpr static std::array<T, 14> trigonSinCoeffs = 
    {0.569230686359505514690621199372, -0.666916672405979070780437163480, 
    0.104282368734236949480920252186, -0.00684063353699157900985137450329, 
    0.000250006884950386227652215859008, -5.85024830863914369171711619397*1e-6, 
    9.53477275029940114004406775030*1e-8, -1.14563844170946315134756461815*1e-9, 
    1.05742726175391285886989821647*1e-11, -7.73527099540430709415664628627*1e-14, 
    4.59595614618295945919569164343*1e-16, -2.26230592819741110431266043618*1e-18, 
    9.37764779915313579625162444173701470571*1e-21, -3.3185994916985649095153565522258013*1e-23};
// clang-format on

template <typename T, std::uint32_t nExpansionTerms = 14>
Vec<T> sin(Vec<T> const &x) {
    Vec<T> sum = hw::Set(d<T>, T(0.0));
    Vec<T> xOverPi = hw::Mul(x, hw::Set(d<T>, INVERSE_PI<T>));

    Vec<T> T2n = hw::Set(d<T>, T(1.0));
    Vec<T> T2np1 = xOverPi;
    for (std::uint32_t n = 0; n < nExpansionTerms; n++) {
        // Sum over odd chebyshev polynomials.
        Vec<T> coeff = hw::Set(d<T>, trigonSinCoeffs<T>[n]);
        sum = hw::MulAdd(coeff, T2np1, sum);

        T2n = chebyshevNext<T>(T2np1, T2n, xOverPi);
        T2np1 = chebyshevNext<T>(T2n, T2np1, xOverPi);
    }
    return sum;
}

template <typename T>
__m128 arcsin(Vec<T> const &x) {
    // This implementation is using the following identity:
    // arcsin(x)=π/2−arcsin(sqrt(1−x^2))
    // This equation divides the definition area into two parts defined by
    //  x = sqrt(1-x^2) -> x = sqrt(2)/2
    // We compute arcsin for -sqrt(2) / 2 <= x <= sqrt(2)/2.
    // Outside this area we use above identity.

    Vec<T> ones = hw::Set(d<T>, T(1.0));
    Vec<T> twos = hw::Set(d<T>, T(2.0));
    Vec<T> halfSqrt2 = hw::Set(d<T>, T(0.70710678118654752440));

    // Check if any element greater than sqrt(2)/2.
    auto mask1 = hw::Gt(x, halfSqrt2);
    // Check if any element smaller than than -sqrt(2)/2.
    auto mask2 = hw::Lt(x, hw::BitCast<>(d<T>, hw::Xor(halfSqrt2, SIGNMASK)));
    // Bitwise or. True for any element smaller than -sqrt(2)/2 or greater than sqrt(2)/2.
    auto combinedMask = hw::Or(mask1, mask2);
    // Compute 1-x^2
    Vec<T> x2 = _hw::Mul(x, x);
    Vec<T> sqrtt = hw::Sqrt(hw::Sub(ones, x2));
    // This is the effective vector for which we calculate arcsin using taylor expansion.
    Vec<T> xMod = hw::IfThenElse(combinedMask, sqrtt, x);
    Vec<T> x2Mod = hw::Mul(xMod, xMod);

    // Coefficients
    Vec<T> dn = ones;
    Vec<T> cn = twos;

    // Argument power.
    Vec<T> sum = xMod;
    Vec<T> nextX = hw::Mul(x2Mod, xMod);
    Vec<T> counter2 = twos;
    Vec<T> i_vec = hw::Set(d<T>, T(3.0));

    // Determine the number of iterations needed
    auto absValueVec = hw::BitCast(d<T>, hw::AndNot(SIGNMASK, t_mod));
    T maxElement = hw::MaxOfLanes(absValueVec);

    std::uint32_t maxIter = std::ceil(-5 / std::log10(std::fabs(maxElement)));

    for (int i = 3; i <= maxIter; i += 2) {
        auto factor = hw::Div(dn, _mm_mul_ps(cn, i_vec));
        sum = hw::MulAdd(factor, next_t, sum);

        next_t = hw::Mul(next_t, x2Mod);
        dn = hw::Mul(dn, i_vec);

        counter2 = hw::Add(counter2, twos);
        cn = hw::Mul(cn, counter2);

        i_vec = hw::Add(i_vec, twos);
    }
    static const __m128 pi2 = hw::Set(d<T>, PI_2<float>);
    auto sol1 = hw::Sub(pi2, sum);
    auto sol2 = hw::Sub(sum, pi2);

    return hw::IfThenElse(mask2, sol2, hw::IfThenElse(mask1, sol1, sum));
}

} // namespace trigon