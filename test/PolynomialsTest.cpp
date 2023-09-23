#include "TestCommon.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <CommonUtils.h>
#include <Polynomials.h>
#include <Sin.h>

namespace trigon::test {

template <typename T>
T chebyshevScalar(std::uint32_t n, T x) {

    if (n == 0) {
        return T(1.0);
    }
    if (n == 1) {
        return x;
    }
    T tn1 = T(1.0);
    T tn = x;
    T tnp1;
    for (std::uint32_t i = 2; i <= n; i++) {
        tnp1 = T(2.0) * x * tn - tn1;
        tn1 = tn;
        tn = tnp1;
    }
    return tn;
}

template <typename T>
void chebyshevTest() {

    std::size_t laneCount = HWY_MAX_LANES_D(D<T>);

    std::vector<T> testCase(laneCount, 0.1);
    T factor = 1.0;
    std::transform(testCase.begin(), testCase.end(), testCase.begin(), [&](auto a) {
        return a * (factor++);
    });
    for (std::uint32_t n = 0; n < 7; n++) {
        std::vector<T> expectedValues = testCase;
        std::transform(expectedValues.begin(),
                       expectedValues.end(),
                       expectedValues.begin(),
                       [&](auto const &a) { return chebyshevScalar<T>(n, a); });

        auto testFunc = [&](Vec<T> const &x) { return chebyshev<T>(n, x); };

        runTestCase<T>(testCase, expectedValues, testFunc);
    }
}

TEST(PolynomialsTest, Chebyshev) {
    chebyshevTest<float>();
    chebyshevTest<double>();
}

template <typename T>
void chebyshevNextTest() {

    std::size_t laneCount = HWY_MAX_LANES_D(D<T>);
    std::vector<T> testCase(laneCount);
    std::iota(testCase.begin(), testCase.end(), 0);
    std::transform(testCase.begin(), testCase.end(), testCase.begin(), [](auto const &a) {
        return T(0.1) * a;
    });

    Vec<T> testVec = LoadU(d<T>, testCase.data());
    // First chebyshev polynomial.
    Vec<T> t0 = hw::Set(d<T>, T(1.0));
    // Second Chebyshev polynomial.
    Vec<T> t1 = testVec;

    for (std::uint32_t n = 2; n < 10; n++) {
        std::vector<T> expectedValues = testCase;
        std::transform(expectedValues.begin(),
                       expectedValues.end(),
                       expectedValues.begin(),
                       [&](auto const &a) { return chebyshevScalar<T>(n, a); });
        Vec<T> t2 = ::trigon::chebyshevNext<T>(t1, t0, testVec);
        std::vector<T> trigonResult(laneCount);
        hw::StoreU(t2, d<T>, trigonResult.data());
        if constexpr (std::is_same_v<T, float>) {
            EXPECT_THAT(expectedValues,
                        ::testing::Pointwise(::testing::FloatNear(1e-7F), trigonResult));
        } else {

            EXPECT_THAT(expectedValues,
                        ::testing::Pointwise(::testing::DoubleNear(1e-8), trigonResult));
        }
        t0 = t1;
        t1 = t2;
    }
}

TEST(PolynomialsTest, ChebyshevNext) {
    chebyshevNextTest<float>();
    chebyshevNextTest<double>();
}

} // namespace trigon::test