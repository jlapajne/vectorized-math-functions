add_library(trigon INTERFACE
    CommonUtils.H
    Sin.h
    Cos.h
    Tan.h
    )

target_link_libraries(trigon INTERFACE hwy::hwy)