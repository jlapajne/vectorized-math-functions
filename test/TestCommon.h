#pragma once

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

template <typename T, typename TestFunc>
void runTestCase(std::vector<T> testValues,
                 std::vector<T> expectedValues,
                 TestFunc funcToTest,
                 T Tolerance = EPS<T>) {
    std::size_t laneCount = HWY_MAX_LANES_D(D<T>);
    Vec<T> testVec = LoadU(d<T>, testValues.data());
    Vec<T> s = funcToTest(testVec);
    std::vector<T> trigonResult(laneCount);
    hw::StoreU(s, d<T>, trigonResult.data());

    if constexpr (std::is_same_v<T, float>) {
        EXPECT_THAT(expectedValues,
                    testing::Pointwise(::testing::FloatNear(Tolerance), trigonResult));
    } else {
        EXPECT_THAT(expectedValues,
                    testing::Pointwise(::testing::DoubleNear(Tolerance), trigonResult));
    }
};
} // namespace trigon::test