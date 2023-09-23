#pragma once

#include "CommonUtils.h"

#include <hwy/highway.h>

#include <array>
#include <immintrin.h>
#include <vector>

#include <Polynomials.h>

namespace trigon {
#pragma clang optimize off
template <typename T>
    requires std::is_floating_point_v<T>
inline static constexpr std::array<T, 12> trigonCosCoeffs = {
    T(-0.3042421776440938642020349128177049239697),
    T(-0.9708678652630182194109914323663784757039),
    T(0.3028491552626994215074191186309676140775),
    T(-0.02909193396501112114732073920800360778849),
    T(0.001392243991176231859984622208952274539411),
    T(-0.00004018994451075494298816526236368837878949),
    T(7.782767011815306088573057896947073998291 * 1e-7),
    T(-1.082653034185828481093421492678695775590 * 1e-8),
    T(1.135109177911507701030194019523024834037 * 1e-10),
    T(-9.295296632678756552885410084526215786661 * 1e-13),
    T(6.111364188334767723806229076684641965132 * 1e-15),
    T(-3.297657841343458986382435554107381460019 * 1e-17)};

template <typename T>
    requires std::is_floating_point_v<T>
inline Vec<T> cos(Vec<T> const &x) {

    Vec<T> sum = hw::Set(d<T>, T(trigonCosCoeffs<T>[0]));
    Vec<T> xOverPi = hw::Mul(x, hw::Set(d<T>, INVERSE_PI<T>));

    Vec<T> T2n = hw::Set(d<T>, T(1));
    Vec<T> T2np1 = xOverPi;

    for (int n = 1; n < 9; n++) {
        Vec<T> coeff = hw::Set(d<T>, trigonCosCoeffs<T>[n]);

        T2n = chebyshevNext<T>(T2np1, T2n, xOverPi);
        T2np1 = chebyshevNext<T>(T2n, T2np1, xOverPi);
        sum = hw::MulAdd(coeff, T2n, sum);
    }
    return sum;
}

// Arccos is define on an interval [-1, 1] and it's image lies in the interval [0, Pi/2].

template <typename T>
    requires std::is_floating_point_v<T>
Vec<T> arccos(Vec<T> const &x) {
    // This function is using Taylor series expansion for ArcCos[1-x].

    // Coefficients in Taylor expansion.
    constexpr static std::array<T, 29> coeffs{
        0.11785113019775792073,        0.026516504294495532165,
        0.0078918167543141464777,      0.0026854098677874526209,
        0.00098871908768538028314,     0.00038344554362157376365,
        0.00015429118302868087157,     0.000063815287098259551659,
        0.000026962891771048260862,    0.000011587623725414788299,
        5.0495474929920174306 * 1e-6,  2.2260088531606476840 * 1e-6,
        9.9092274446253903312 * 1e-7,  4.4481692162142300194 * 1e-7,
        2.0112421026000900249 * 1e-7,  9.1515324838952959939 * 1e-8,
        4.1874028886394862762 * 1e-8,  1.9255137156844484566 * 1e-8,
        8.8934827151552292074 * 1e-9,  4.1240814663875315928 * 1e-9,
        1.9193191985042748083 * 1e-9,  8.9616696920060710114 * 1e-10,
        4.1968966527086710911 * 1e-10, 1.9708640956278602127 * 1e-10,
        9.2785190070637105306 * 1e-11, 4.3783432397265441020 * 1e-11,
        2.0704993536013236334 * 1e-11, 9.8108717804573997353 * 1e-12,
        4.6574404463334441467 * 1e-12};

    // Get the absolute value of the vector;
    Vec<T> x_abs = hw::AndNot(SignMask<T>, x);
    Vec<T> sqrt2 = hw::Set(d<T>, T(1.4142135623730950488));

    Vec<T> t = hw::Sub(hw::Set(d<T>, T(1)), x_abs);
    Vec<T> sqrtt = hw::Sqrt(t);

    std::vector<Vec<T>> powersOfInput;
    powersOfInput.reserve(coeffs.size());
    auto *lastPower = &sqrtt;
    for (auto const &coeff : coeffs) {
        lastPower = &powersOfInput.emplace_back(hw::Mul(t, *lastPower));
    }

    Vec<T> result = hw::Mul(sqrt2, sqrtt);

    for (std::uint32_t i = 0; i < coeffs.size(); i++) {

        result = hw::MulAdd(hw::Set(d<T>, coeffs[i]), powersOfInput[i], result);
    }

    Vec<T> neg_result = hw::Add(hw::Set(d<T>, PI<T>), hw::Xor(result, hw::SignBit(d<T>)));

    auto mask = hw::Gt(x, hw::Zero(d<T>));
    return hw::IfThenElse(mask, result, neg_result);
}
} // namespace trigon