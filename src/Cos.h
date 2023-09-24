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

    for (int n = 1; n < trigonCosCoeffs<T>.size(); n++) {
        Vec<T> coeff = hw::Set(d<T>, trigonCosCoeffs<T>[n]);

        T2n = chebyshevNext<T>(T2np1, T2n, xOverPi);
        T2np1 = chebyshevNext<T>(T2n, T2np1, xOverPi);
        sum = hw::MulAdd(coeff, T2n, sum);
    }
    return sum;
}

// Arccos is define on an interval [-1, 1] and it's image lies in the interval [0, Pi].
template <typename T>
    requires std::is_floating_point_v<T>
Vec<T> arccos(Vec<T> const &x) {
    // This function is using Taylor series expansion for ArcCos[1-x].

    // Coefficients in Taylor expansion.
    constexpr static std::array<T, 49> coeffs{
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
        4.6574404463334441467 * 1e-12, 2.2148292614326118955 * 1e-12,
        1.0549641169727021074 * 1e-12, 5.0326353128180945484 * 1e-13,
        2.4042157617205392884 * 1e-13, 1.1500985245485401605 * 1e-13,
        5.5086711019875248531 * 1e-14, 2.6416677154793676545 * 1e-14,
        1.2682384915125720929 * 1e-14, 6.0952166052274590076 * 1e-15,
        2.9323709227842911762 * 1e-15, 1.4121085593438858974 * 1e-15,
        6.8063798544337609265 * 1e-16, 3.2835539787951105758 * 1e-16,
        1.5853834199942979090 * 1e-16, 7.6607297663028861550 * 1e-17,
        3.7045568057927448861 * 1e-17, 1.7927439755008018000 * 1e-17,
        8.6816588152891572050 * 1e-18, 4.2070431060988318178 * 1e-18,
        2.0399952888725988752 * 1e-18};

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