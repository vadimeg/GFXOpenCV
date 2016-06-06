#include <opencv2/opencv.hpp>

namespace cv {
namespace igpu {

bool boxFilter( InputArray _src, OutputArray _dst, int ddepth,
                Size ksize, Point anchor, bool normalize, int borderType);

}
}
