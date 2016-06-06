#pragma once

#include "../../../core/src/igpu/build_list.hpp"
#include "precomp.hpp"

#define CV_IGPU_RUN(condition, func)              \
    if ((condition) && func)                      \
        return;

namespace cv {
namespace igpu {

//struct BilateralGpuTask {
//    bool async;
//
//    Offload2dBuffer src;
//    Offload2dBuffer dst;
//
//    Offload2dBuffer space;
//    
//    using LayoutT = OpLayoutsServer::PrimeLayoutT;
//    LayoutT layoutGpu;
//    LayoutT layoutCpu;
//    
//    int cn;
//
//    int d;
//    float gauss_color_coeff;
//    float gauss_space_coeff;
//};
//
//
//class BilateralFilterIgpuWorker {
//public:
//    BilateralFilterIgpuWorker();
//    
//    void start(const BilateralGpuTask& _task);
//    
//    void finalize();
//};


bool bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType );

}
}
