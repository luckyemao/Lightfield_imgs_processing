# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("light_field_yem")


find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})

message(STATUS "OpenCL_LIBRARIES " ${OpenCL_LIBRARIES} )
message(STATUS "OpenCV_LIBS "  ${OpenCV_LIBRARIES})
message(STATUS "PROJECT_SOURCE_DIR " ${PROJECT_SOURCE_DIR})
message(STATUS "PROJECT_BINARY_DIR " ${PROJECT_BINARY_DIR})

# 包含子项目。
add_subdirectory (light_field_yem)
add_subdirectory(calibration)
add_subdirectory(render)
add_subdirectory(cameraio)
add_subdirectory(Polarization_based_imaging)
add_subdirectory (refocus_block)
add_subdirectory (testenv)
add_subdirectory (eDOF)
add_subdirectory (VideoRender)
add_subdirectory (DisparityCompute)