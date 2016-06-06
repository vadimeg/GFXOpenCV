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
using namespace std::chrono;

const int timeDivider = 10000;
const int epsilonDiff = 5;
const string correctnessCheckerResult = "Correctness: ";

void correctnessChecker(const Mat &test_img, int imgNumber, int kernelSize) {
    string img_name = "test_img\\" + string("morph") + std::to_string(kernelSize) + "_" + std::to_string(imgNumber) + "_original.bmp";
    Mat original = imread(img_name, CV_LOAD_IMAGE_COLOR);
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
        imwrite(string("morph") + std::to_string(kernelSize) + "_" + std::to_string(imgNumber) + "_testfailed.bmp", dst);
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
    int kernelSize[] = {3, 5, 7, 9, 11, 13};
    Mat A;
    Mat B;
    
    for (int i = 0; i < 3; i++) {
        std::cout << "Picture #" << i << std::endl;
        A = imread(imgpath[i], CV_LOAD_IMAGE_COLOR);
        for (int j = 0; j < 6; j++) {
            std::cout << "Kernel size = " << kernelSize[j] << std::endl;
            std::cout << "Performance test:" << std::endl;
            Mat kernel = Mat::ones(kernelSize[j], kernelSize[j], CV_8U);
            cv::erode(A, B, kernel);
            measuredTime(&cv::erode, A, B, kernel, cv::Point(-1, -1), -1, 0, morphologyDefaultBorderValue()); 
            
            correctnessChecker(B, i, kernelSize[j]);
            std::cout << std::endl;
        }
    }

    return 0;
}