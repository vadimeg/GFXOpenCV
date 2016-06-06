set(the_description "Image Processing")
ocv_define_module(imgproc opencv_core WRAP java python)

if(INTEL_OFFLOAD)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qoffload:optional")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qgpu-arch=ivybridge")
endif()