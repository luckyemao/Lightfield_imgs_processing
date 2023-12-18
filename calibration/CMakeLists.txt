
#add_library(calibration calibration.cpp)

#add_executable(estimate_distance estimate_distance.cpp)
#target_link_libraries(estimate_distance ${OpenCV_LIBS})

add_executable(LFCalibrate LFCalibrate.cpp "calibration.h"  )
target_link_libraries(LFCalibrate ${OpenCL_LIBRARIES} ${OpenCV_LIBS} Camera)

configure_file(kernel.ocl kernel.ocl COPYONLY)
configure_file(../LF.h LF.h COPYONLY)
#configure_file(input output) //"input" is cmake files current position ;"output" is runing position  实际上就是在运行前就将需要的文件拷贝到需要的位置 cmake真的太强了
#参见https://www.jianshu.com/p/2946eeec2489 博客

#add_executable(KernelTest KernelTest.cpp)
#target_link_libraries(KernelTest ${OpenCL_LIBRARIES} ${OpenCV_LIBS})
