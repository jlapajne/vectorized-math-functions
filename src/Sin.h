#pragma once

#include "CommonUtils.h"

#include <hwy/highway.h>

#include <array>
#include <immintrin.h>

#include <Polynomials.h>

#pragma optimize("", off)
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
    requires(2 <= nExpansionTerms && nExpansionTerms <= trigonSinCoeffs<T>.size())
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
struct ArcSinPadeSeriesCoeffs {
    // clang-format off
    constexpr static std::array<T, 8> nominatorCoeffs{
        T(-4.02134331227440014268247941231),
        T(6.64013839618028461107282195693),
        T(-5.80083406159300799625908304343),
        T(2.87770102416224398385250178038),
        T(-0.807703699415387251626727034418),
        T(0.120037918234026234044753083504),
        T(-0.00802485128856183220648189624477),
        T(0.000158745886079939725898992545678)};
    
    constexpr static std::array<T, 9> denominatorCoeffs{
        T(-4.18800997894106680934914607898),
        T(7.26314005933712907929767963676),
        T(-6.74189951353813997496465321744),
        T(3.61319822539326438446745092080),
        T(-1.12364053678773054096367702728),
        T(0.192972874306975617965576491510),
        T(-0.0161688505730104310006288406829),
        T(0.000495519108487209229203679877131),
        T(-2.01848846132919200654943379374 * 1e-6)};
    // clang-format on
};

template <>
struct ArcSinPadeSeriesCoeffs<float> {
    // clang-format ooff
    inline constexpr static std::array<float, 5> nominatorCoeffs{
        -2.52049062703774555974805954944f,
        2.29364992492967237163510606982f,
        -0.906289894356850270625703003915f,
        0.145239679370744583225485028826f,
        -0.00653499178782165635075056473459f};

    inline constexpr static std::array<float, 6> denominatorCoeffs{
        -2.68715729370441222641472621610f,
        2.66650947388040774270422710583f,
        -1.19381420011861112028577924582f,
        0.233800936921604100138415608015f,
        -0.0157374465918093307096312958063f,
        0.000134117645146253114389103579422f};
    // clang-format on
};

// ArcSin using pade approximation
template <typename T>
    requires(std::is_floating_point_v<T>)
Vec<T> arcsin(Vec<T> const &x) {
    // This implementation is using the following identity:
    // arcsin(x)=π/2−arcsin(sqrt(1−x^2))
    // This equation divides the definition area into two parts defined by
    //  x = sqrt(1-x^2) -> x = sqrt(2)/2
    // We compute arcsin for -sqrt(2) / 2 <= x <= sqrt(2)/2.
    // Outside this area we use above identity. For the actual arcsin clculation
    // we use pade approximation

    auto const &nominatorCoeffs = ArcSinPadeSeriesCoeffs<T>::nominatorCoeffs;
    auto const &denominatorCoeffs = ArcSinPadeSeriesCoeffs<T>::denominatorCoeffs;

    Vec<T> halfSqrt2 = hw::Set(d<T>, T(0.70710678118654752440));

    // Check if any element greater than sqrt(2)/2.
    auto mask1 = hw::Gt(x, halfSqrt2);
    // Check if any element smaller than than -sqrt(2)/2.
    auto mask2 = hw::Lt(x, hw::BitCast<>(d<T>, hw::Xor(halfSqrt2, SignMask<T>)));
    // Bitwise or. True for any element smaller than -sqrt(2)/2 or greater than sqrt(2)/2.
    auto combinedMask = hw::Or(mask1, mask2);

    // Compute 1-x^2
    Vec<T> x2 = hw::Mul(x, x);
    Vec<T> minusOne = hw::Set(d<T>, T(-1.0));
    Vec<T> sqrtt = hw::Sqrt(hw::Mul(minusOne, hw::MulAdd(x, x, minusOne)));
    // This is the effective vector for which we calculate arcsin using taylor expansion.
    Vec<T> xMod = hw::IfThenElse(combinedMask, sqrtt, x);
    Vec<T> x2Mod = hw::Mul(xMod, xMod);

    Vec<T> nominator = xMod;
    Vec<T> denominator = hw::Set(d<T>, T(1));

    Vec<T> oddPower = xMod;
    Vec<T> evenPower = x2Mod;

    for (std::uint32_t i = 0; i < nominatorCoeffs.size(); i++) {
        oddPower = hw::Mul(oddPower, x2Mod);
        nominator = hw::MulAdd(hw::Set(d<T>, nominatorCoeffs[i]), oddPower, nominator);

        denominator = hw::MulAdd(hw::Set(d<T>, denominatorCoeffs[i]), evenPower, denominator);
        evenPower = hw::Mul(evenPower, x2Mod);
    }

    if constexpr (denominatorCoeffs.size() > nominatorCoeffs.size()) {
        denominator =
            hw::MulAdd(hw::Set(d<T>, denominatorCoeffs.back()), evenPower, denominator);
    }

    Vec<T> arcsinVec = hw::Div(nominator, denominator);

    Vec<T> pi2 = hw::Set(d<T>, PI_2<T>);
    auto sol1 = hw::Sub(pi2, arcsinVec);
    auto sol2 = hw::Sub(arcsinVec, pi2);

    return hw::IfThenElse(mask2, sol2, hw::IfThenElse(mask1, sol1, arcsinVec));
}

} // namespace trigon