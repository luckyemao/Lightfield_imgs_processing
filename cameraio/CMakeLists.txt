set(HikSdkRoot "../3rd")

include_directories("${HikSdkRoot}/Includes")
add_library(Camera Camera.cpp)
target_link_libraries(Camera ${OpenCV_LIBS} MvCameraControl)
target_link_directories(Camera PUBLIC "${HikSdkRoot}/Libraries/Win64/")

