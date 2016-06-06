#pragma once

#include "precomp.hpp"
#include "../../../core/src/igpu/build_list.hpp"

#define CV_IGPU_RUN(condition, func)              \
    if ((condition) && func)                      \
        return;

namespace cv {
namespace igpu {

bool Canny( InputArray _src, OutputArray _dst,
               double low_thresh, double high_thresh,
               int aperture_size, bool L2gradient );

}
}
 