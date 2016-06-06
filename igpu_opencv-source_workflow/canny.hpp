#pragma once

#include <opencv2/opencv.hpp>

namespace cv {
namespace igpu {

bool Canny( InputArray _src, OutputArray _dst,
               double low_thresh, double high_thresh,
               int aperture_size, bool L2gradient );

}
}
 