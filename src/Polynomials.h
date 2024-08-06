#pragma once

#include <hwy/highway.h>

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <span>
#include <string>
#include <vector>

namespace trigon {

namespace hw = hwy::HWY_NAMESPACE;

/**
 * Calculate Chebyshev polynomial of degree n at the given x values.
 *
 * @param n the degree of the Chebyshev polynomial
 * @param x the vector of x values
 *
 * @return the vector of Chebyshev polynomial values.
 */
template <typename T>
Vec<T> chebyshev(const int &n, Vec<T> const &x) {
    Vec<T> t0 = hw::Set(d<T>, T(1.0));
    if (n == 0)
        return t0;
    if (n == 1)
        return x;

    Vec<T> twos = hw::Set(d<T>, T(2.0));
    Vec<T> twiceX = hw::Set(d<T>, T(2.0)) * x;

    // T2
    Vec<T> t2 = hw::MulSub(twiceX, x, t0);
    if (n == 2)
        return t2;

    // T3
    Vec<T> t3 = hw::MulSub(twiceX, t2, x);
    if (n == 3)
        return t3;

    Vec<T> Tn = t3;
    Vec<T> Tnm1 = t2;
    Vec<T> Tnp1;
    for (int i = 4; i <= n; i++) {
        Tnp1 = hw::MulSub(hw::Mul(twos, Tn), x, Tnm1);
        Tnm1 = Tn;
        Tn = Tnp1;
    }
    return Tnp1;
}

/**
 * Calculate the next Chebyshev polynomial value using the given coefficients and input values.
 *
 * @param cn the current coefficient vector
 * @param cn_1 the previous coefficient vector
 * @param x the input vector
 *
 * @return the next Chebyshev polynomial value
 */
template <typename T>
Vec<T> chebyshevNext(Vec<T> const &cn, Vec<T> const &cn_1, Vec<T> const &x) {
    return hw::MulSub(hw::Mul(hw::Set(d<T>, T(2)), cn), x, cn_1);
}

/**
 * Given two consequtive Legendre polynomials, this code
 * calculates the next Legendre polynomial
 * 
 * @param x_vec vector of x values for which the next polynomial is
 *              calculated calculated
 */
template <typename T>
Vec<T> legendreNext(Vec<T> const &Pn, Vec<T> const &Pnm1, Vec<T> const &x, int const n) {
    Vec<T> n = hw::Set(d<T>, T(n));
    Vec<T> np1 = hw::Set(d<T>, T(n + 1));
    Vec<T> ones = hw::Set(d<T>, T(1));
    Vec<T> twos = hw::Set(d<T>, T(2));
    Vec<T> coeff1 = hw::Div(hw::Mul(hw::MulAdd(twos, n, ones), x), np1);
    return hw::MulSub(coeff1, Pn, hw::Mul(hw::Div(n, np1), Pnm1));
}

} // namespace trigon