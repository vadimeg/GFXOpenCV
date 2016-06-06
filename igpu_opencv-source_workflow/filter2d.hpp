#pragma once

#include <opencv2/opencv.hpp>

namespace cv {
namespace igpu {

bool filter2D( InputArray _src, OutputArray _dst, int ddepth,
               InputArray _kernel, Point anchor0,
               double delta, int borderType );

}
}
 