add_executable(trigon_test
TrigonometricFunctionsTest.cpp
PolynomialsTest.cpp
TestData.h
TestCommon.h
)

target_link_libraries(trigon_test trigon GTest::gtest_main)

add_test(NAME trigon_test COMMAND trigon_test)