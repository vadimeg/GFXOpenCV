#pragma once

#include <opencv2/opencv.hpp>

namespace cv {
namespace igpu {

bool bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType );

}
}
