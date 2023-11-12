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

// These type and tag definition match dose in highway documentation.
template <typename T>
using D = hw::ScalableTag<T>;

template <typename T>
inline static constexpr D<T> d;

template <typename T>
using Vec = hw::Vec<D<T>>;

template <typename T>
    requires std::is_floating_point_v<T>
inline constexpr static T PI =
    T(3.1415926535897932384626433832795028841971693993751058209749445923);

template <typename T>
    requires std::is_floating_point_v<T>
inline constexpr static T PI_2 =
    T(1.5707963267948966192313216916397514420985846996875529104874722961);

template <typename T>
    requires std::is_floating_point_v<T>
inline constexpr static T INVERSE_PI = T(0.31830988618379067153776752674502872406891929148091);

template <typename T>
inline static Vec<T> const SignMask = hw::Set(d<T>, T(-0.0));

template <typename T>
std::string vecToString(Vec<T> const &vec) {

    std::string to_return;
    for (std::size_t i = 0; i < hw::Lanes(d<T>); i++) {
        to_return += std::to_string(ExtractLane(vec, i)) + " ";
    }
    return to_return;
}

} // namespace trigon