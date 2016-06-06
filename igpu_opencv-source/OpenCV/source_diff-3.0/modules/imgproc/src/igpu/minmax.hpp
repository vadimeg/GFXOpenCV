#pragma once

#include "precomp.hpp"
#include "../../../core/src/igpu/build_list.hpp"

#define CV_IGPU_RUN(condition, func)              \
    if ((condition) && func)                      \
        return;

namespace cv {
namespace igpu {

bool morphOp( int op, InputArray _src, OutputArray _dst, InputArray _kernel,
                          Point anchor, int iterations,
                          int borderType,
                          const Scalar& borderValue );

}
}
