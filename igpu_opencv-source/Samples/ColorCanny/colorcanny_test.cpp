#include <opencv2\imgproc.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <iostream>
#include <sstream>
#include <chrono>
#include <ratio>
#include <intrin.h>
#pragma intrinsic(__rdtsc)

static unsigned __int64 rdtsc() {
    return __rdtsc();
}

using namespace cv;
using namespace std;

const int timeDivider = 10000;
const int epsilonDiff = 5;
const string correctnessCheckerResult = "Correctness: ";

void correctnessChecker(const Mat &test_img, int imgNumber, bool normType) {
    string img_name = "test_img\\canny_color_" + std::to_string(imgNumber) + ((normType) ? ("_true") : ("_false")) + "_original.bmp";
    Mat original = imread(img_name, CV_LOAD_IMAGE_GRAYSCALE);
    Mat dst;
    bool testStatus = true;

    if (original.empty()) {
        std::cout << correctnessCheckerResult + "Can't open original file!" << std::endl;
        return;    
    }

    absdiff(original, test_img, dst);
    for (int i = 0; i < dst.rows; i++) {
        if (!testStatus) 
            break;

        unsigned char* row = dst.ptr<unsigned char>(i);
        for (int j = 0; j < dst.cols; j++) {
            if (row[j] > epsilonDiff) {
                testStatus = false;
                break;
            }
        }
    }

    if (testStatus) 
        std::cout << correctnessCheckerResult + "[OK]" << std::endl;
    else {
        std::cout << correctnessCheckerResult + "[FAILED]" << std::endl;
        imwrite(string("canny_color_") + std::to_string(imgNumber) + ((normType) ? ("_true") : ("_false")) + "_testfailed.bmp", dst);
    }
}

template <class testFunc, typename... Args>
void measuredTime(const testFunc func, Mat &A, Mat &B, Args ... taskArgs) {
    auto mintime = rdtsc();

    for (int i = 0; i < 50; i++) {
        auto start = rdtsc();
        func(A, B, taskArgs...);
        auto finish = rdtsc();

        if (!i)
            mintime = finish - start;
        else if (finish - start < mintime)
            mintime = finish - start;
    }

    std::cout << "Time " << (mintime / timeDivider) << " units" << std::endl;
}

int main() {
    string imgpath[] = {"test_img\\1.jpg", "test_img\\2.jpg", "test_img\\3.jpg"};
    Mat src;
    Mat B;
    Mat dst;

    for (int i = 0; i < 3; i++) {
        std::cout << "Picture #" << i << std::endl;
        src = imread(imgpath[i], CV_LOAD_IMAGE_COLOR);

        std::cout << "Performance test:" << std::endl;
        std::cout << "First norm type: ";
        measuredTime(&cv::Canny, src, dst, 50.0, 50.0, 3, false);
        correctnessChecker(dst, i, false);
        
        std::cout << "Second norm type: ";
        measuredTime(&cv::Canny, src, dst, 50.0, 50.0, 3, true);
        correctnessChecker(dst, i, true);
    }

    return 0;
}