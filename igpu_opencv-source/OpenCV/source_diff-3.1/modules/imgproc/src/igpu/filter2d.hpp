#pragma once

#include "precomp.hpp"
#include "../../../core/src/igpu/build_list.hpp"

#define CV_IGPU_RUN(condition, func)              \
    if ((condition) && func)                      \
        return;

namespace cv {
namespace igpu {

bool filter2D( InputArray _src, OutputArray _dst, int ddepth,
               InputArray _kernel, Point anchor0,
               double delta, int borderType );

bool sepFilter2D( InputArray _src, OutputArray _dst, int ddepth,
               InputArray _kernelX, InputArray _kernelY, Point anchor0,
               double delta, int borderType );

}
}
 