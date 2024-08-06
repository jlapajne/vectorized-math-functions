#include "TestCommon.h"
#include "TestData.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hwy/highway.h>

#include <CommonUtils.h>
#include <Cos.h>
#include <Sin.h>

namespace trigon::test {

template <typename T, TestFuncKind Kind>
struct TestFuncSturct {};

template <typename T>
struct TestFuncSturct<T, TestFuncKind::Sin> {
    constexpr static std::string_view name{"sin"};
    constexpr static auto TestFunc = sin<T>;
    constexpr static T (*RefFunc)(T) = std::sin;
};

template <typename T>
struct TestFuncSturct<T, TestFuncKind::ArcSin> {
    constexpr static std::string_view name{"arcSin"};
    constexpr static auto TestFunc = arcsin<T>;
    constexpr static T (*RefFunc)(T) = std::asin;
};

template <typename T>
struct TestFuncSturct<T, TestFuncKind::Cos> {
    constexpr static std::string_view name{"cos"};
    constexpr static auto TestFunc = cos<T>;
    constexpr static T (*RefFunc)(T) = std::cos;
};

template <typename T>
struct TestFuncSturct<T, TestFuncKind::ArcCos> {
    constexpr static std::string_view name{"arcCos"};
    constexpr static auto TestFunc = arccos<T>;
    constexpr static T (*RefFunc)(T) = std::acos;
};

template <typename T, TestFuncKind Kind>
constexpr static auto TestFunc = TestFuncSturct<T, Kind>::TestFunc;

template <typename T, TestFuncKind Kind>
constexpr static auto RefFunc = TestFuncSturct<T, Kind>::RefFunc;

template <typename T, TestFuncKind Kind>
void validateAgainstMathematica() {
    std::size_t laneCount = HWY_MAX_LANES_D(D<T>);

    constexpr T intervalStart = testInterval<T, Kind>.min;
    constexpr T intervalEnd = testInterval<T, Kind>.max;
    std::size_t testDataSize = accurateData<T, Kind>.size();
    T step = (intervalEnd - intervalStart) / T(testDataSize - 1);

    constexpr T Tolerance =
        []() -> std::conditional_t<std::is_same_v<T, double>, double, float> {
        if constexpr (std::is_same_v<T, double>)
            return 1e-14;
        return 1e-6f;
    }();

    T maxUlpError{};
    for (std::uint32_t i = 0; i < testDataSize; i += laneCount) {
        std::vector<T> testCase(laneCount);
        for (std::uint32_t j = 0; j < laneCount; j++) {
            testCase[j] = intervalStart + (i + j) * step;
        }

        auto startIt = accurateData<T, Kind>.begin() + i;
        std::vector<T> expectedValues(startIt, startIt + laneCount);

        std::vector<T> ulpDiff =
            runTestCase<T>(testCase, expectedValues, TestFunc<T, Kind>, Tolerance);
        maxUlpError = std::max(maxUlpError, *std::ranges::max_element(ulpDiff));
    }

    std::cout << std::format("max ulp error = {:g}\n", maxUlpError);
}

TEST(TrigonometricTest, SinVsMathematicaFloat) {
    validateAgainstMathematica<float, TestFuncKind::Sin>();
}
TEST(TrigonometricTest, SinVsMathematicaDouble) {
    validateAgainstMathematica<double, TestFuncKind::Sin>();
}

TEST(TrigonometricTest, ArcSinVsMathematicaFloat) {
    validateAgainstMathematica<float, TestFuncKind::ArcSin>();
}

TEST(TrigonometricTest, ArcSinVsMathematicaDouble) {
    validateAgainstMathematica<double, TestFuncKind::ArcSin>();
}

TEST(TrigonometricTest, CosVsMathematicaFloat) {
    validateAgainstMathematica<float, TestFuncKind::Cos>();
}

TEST(TrigonometricTest, CosVsMathematicaDouble) {
    validateAgainstMathematica<double, TestFuncKind::Cos>();
}

TEST(TrigonometricTest, ArcCosVsMathematicaFloat) {
    validateAgainstMathematica<float, TestFuncKind::ArcCos>();
}

TEST(TrigonometricTest, ArcCosVsMathematicaDouble) {
    validateAgainstMathematica<double, TestFuncKind::ArcCos>();
}

template <typename T, TestFuncKind Kind>
void validateAgainstStd() {

    std::size_t laneCount = HWY_MAX_LANES_D(D<T>);
    constexpr std::uint32_t testPointsCount = 256;

    constexpr T intervalStart = testInterval<T, Kind>.min;
    constexpr T intervalEnd = testInterval<T, Kind>.max;
    T step = (intervalEnd - intervalStart) / T(testPointsCount - 1);

    constexpr T Tolerance =
        []() -> std::conditional_t<std::is_same_v<T, double>, double, float> {
        if constexpr (std::is_same_v<T, double>)
            return 1e-14;
        return 1e-6f;
    }();

    T maxUlpError{};
    for (std::uint32_t i = 0; i < testPointsCount; i += laneCount) {
        std::vector<T> testCase(laneCount);
        for (std::uint32_t j = 0; j < laneCount; j++) {
            testCase[j] = intervalStart + (i + j) * step;
        }

        std::vector<T> expectedValues = testCase;
        std::transform(expectedValues.begin(),
                       expectedValues.end(),
                       expectedValues.begin(),
                       [](auto const &a) { return RefFunc<T, Kind>(a); });

        [[maybe_unused]] std::vector<T> ulpDiff =
            runTestCase<T>(testCase, expectedValues, TestFunc<T, Kind>, Tolerance);
        maxUlpError = std::max(maxUlpError, *std::ranges::max_element(ulpDiff));
    }
    std::cout << std::format("max ulp error = {:g}\n", maxUlpError);
}

TEST(TrigonometricTest, SinFloat) { validateAgainstStd<float, TestFuncKind::Sin>(); }
TEST(TrigonometricTest, SinDouble) { validateAgainstStd<double, TestFuncKind::Sin>(); }

TEST(TrigonometricTest, CosFloat) { validateAgainstStd<float, TestFuncKind::Cos>(); }
TEST(TrigonometricTest, CosDouble) { validateAgainstStd<double, TestFuncKind::Cos>(); }

TEST(TrigonometricTest, ArcCos) {
    validateAgainstStd<float, TestFuncKind::ArcCos>();
    validateAgainstStd<double, TestFuncKind::ArcCos>();
}

TEST(TrigonometricTest, ArcSin) {
    validateAgainstStd<float, TestFuncKind::ArcSin>();
    validateAgainstStd<double, TestFuncKind::ArcSin>();
}

} // namespace trigon::test