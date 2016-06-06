#pragma once

#include "../../../core/src/igpu/build_list.hpp"
#include "precomp.hpp"

#define CV_IGPU_RUN(condition, func)              \
    if ((condition) && (func))                      \
        return;

namespace cv {
namespace igpu {

bool boxFilter( InputArray _src, OutputArray _dst, int ddepth,
                Size ksize, Point anchor, bool normalize, int borderType);

}
}
