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

void correctnessChecker(const Mat &test_img, int imgNumber) {
    string img_name = "test_img//" + string("gemm_") + std::to_string(imgNumber) + "_original.bmp";
    Mat _original = imread(img_name, CV_LOAD_IMAGE_GRAYSCALE);
    Mat original;
    Mat dst;
    bool testStatus = true;

    if (_original.empty()) {
        std::cout << correctnessCheckerResult + "Can't open original file!" << std::endl;
        return;    
    }

    _original.convertTo(original, CV_MAKETYPE(CV_32F, 1));
    absdiff(test_img, original, dst);
    for (int i = 0; i < dst.rows; i++) {
        if (!testStatus) 
            break;

        float *row = dst.ptr<float>(i);
        uchar *row_2 = dst.ptr<uchar>(i);
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
        imwrite(string("gemm_") + std::to_string(imgNumber) + "_testfailed.bmp", dst);
    }
}

template <class testFunc, typename... Args>
void measuredTime(const testFunc func, Mat &A, Mat &B, Mat &dst, int flags, Args ... taskArgs) {
    auto mintime = rdtsc();

    for (int i = 0; i < 5; i++) {
        auto start = rdtsc();
        func(A, B, taskArgs..., dst, flags);
        auto finish = rdtsc();

        if (!i)
            mintime = finish - start;
        else if (finish - start < mintime)
            mintime = finish - start;
    }

    std::cout << "Time " << (mintime / timeDivider) << " units" << std::endl;
}

int main() {
    string imgpath[] = {"test_img//1.jpg", "test_img//2.jpg", "test_img//3.jpg"};
    Mat _A;
    Mat A;
    Mat B;
    Mat dst;
    
    for (int i = 0; i < 3; i++) {
        std::cout << "Matrix #" << i << std::endl;
        _A = imread(imgpath[i], CV_LOAD_IMAGE_GRAYSCALE);
        _A.convertTo(A, CV_MAKETYPE(CV_32F, 1));
        B = A.clone().t();

        std::cout << "Performance test:" << std::endl;
        measuredTime(&cv::gemm, A, B, dst, 0, 1, Mat(), 0); 
        correctnessChecker(dst, i);
        std::cout << std::endl;
    }

    return 0;
}