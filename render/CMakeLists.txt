
#add_executable(LFRender LFRender.cpp)
#target_link_libraries(LFRender ${OpenCL_LIBRARIES} ${OpenCV_LIBS} Camera)



add_executable(SubView  "SubView.cpp")
target_link_libraries(SubView ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

#add_executable(LFFullFocus  "LFFullFocus.cpp" )
#target_link_libraries(LFFullFocus ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

#add_executable(ViewDepthTest  "ViewDepthTest.cpp" )
#target_link_libraries(ViewDepthTest ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

#add_executable(OutVideo  "OutVideo.cpp" )
#target_link_libraries(OutVideo ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

#add_executable(LFRender  "LFRender.cpp" )
#target_link_libraries(LFRender ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

#configure_file(../disparityKernel.ocl DisparityKernel.ocl COPYONLY)
#configure_file(RenderKernel.ocl RenderKernel.ocl COPYONLY)

configure_file(../kernels.ocl kernels.ocl COPYONLY)
configure_file(../kernelsbyMao.ocl kernelsbyMao.ocl COPYONLY)
configure_file(../LF.h LF.h COPYONLY)