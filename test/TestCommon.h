#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

#include <CommonUtils.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace trigon::test {

template <typename T>
inline constexpr T EPS{};

template <>
inline constexpr float EPS<float> = 1e-5f;

template <>
inline constexpr double EPS<double> = 1e-8f;

template <std::floating_point T>
T getUlpUnit(T val) {
    int exp;
    T mantissa = std::frexp(val, &exp);
    return std::pow(T(2), exp - 1) * std::pow(T(2), -std::numeric_limits<T>::digits);
}

template <std::floating_point T>
T getULPDifference(T ref, T val) {

    // Usually the mantissa is in range [1, 2). std::frexp returns as if it was in
    // range [0.5, 1) - effectively dividing with 2. Moving 2 out of mantissa means
    // that the exp is multiplied with additional 2, which means we have to divide
    // exp by 2 to get real value of the exponent of the given floating point number.
    return std::abs(ref - val) / getUlpUnit<T>(ref);
}

template <typename T, typename TestFunc>
std::vector<T> runTestCase(std::vector<T> testValues,
                           std::vector<T> expectedValues,
                           TestFunc funcToTest,
                           T Tolerance = EPS<T>) {
    std::size_t laneCount = HWY_MAX_LANES_D(D<T>);
    Vec<T> testVec = LoadU(d<T>, testValues.data());
    Vec<T> s = funcToTest(testVec);
    std::vector<T> trigonResult(laneCount);
    hw::StoreU(s, d<T>, trigonResult.data());

    assert(testValues.size() == expectedValues.size());
    std::vector<T> ulpErrors(expectedValues.size());

    if constexpr (std::is_same_v<T, float>) {
        EXPECT_THAT(expectedValues,
                    testing::Pointwise(::testing::FloatNear(Tolerance), trigonResult));
    } else {
        EXPECT_THAT(expectedValues,
                    testing::Pointwise(::testing::DoubleNear(Tolerance), trigonResult));
    }
    std::transform(expectedValues.begin(),
                   expectedValues.end(),
                   trigonResult.begin(),
                   ulpErrors.begin(),
                   [](T ref, T val) { return getULPDifference(ref, val); });

    return ulpErrors;
};
} // namespace trigon::test