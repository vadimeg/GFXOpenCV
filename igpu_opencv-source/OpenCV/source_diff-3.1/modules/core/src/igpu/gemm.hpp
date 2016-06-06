#pragma once

#include "../../../core/src/igpu/build_list.hpp"
#include "precomp.hpp"

#define CV_IGPU_RUN(condition, func)              \
    if ((condition) && func)                      \
        return;

namespace cv {
namespace igpu {

bool gemm(InputArray src1, InputArray src2, double alpha,
                       InputArray src3, double beta, OutputArray dst, int flags);

}
}
 