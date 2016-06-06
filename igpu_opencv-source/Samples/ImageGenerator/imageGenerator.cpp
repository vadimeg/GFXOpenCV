#include <opencv2\imgproc.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <iostream>
#include <sstream>
#include <chrono>
#include <ratio>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main() {
    string imgpath[] = {"test_img\\1.jpg", "test_img\\2.jpg", "test_img\\3.jpg"};
    int kernelSize[] = {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    Mat A;
    Mat B;
    Mat dst;

    for (int i = 0; i < 3; i++) {
        std::cerr << "Picture #" << i << std::endl;
        A = imread(imgpath[i], CV_LOAD_IMAGE_GRAYSCALE);

        if (A.empty()) {
            std::cerr << "Can't open file " + imgpath[i] + "\n";
            return 1;
        }

        cv::Canny(A, dst, 50.0, 50.0, 3, false);
        imwrite("canny_" + std::to_string(i) + "_false_original.bmp", dst);
        
        cv::Canny(A, dst, 50.0, 50.0, 3, true);
        imwrite("canny_" + std::to_string(i) + "_true_original.bmp", dst);

        // Special type == CV32_FC1 || type == CV_64FC1 || type == CV__32FC1 || type == || CV_64FC2
        Mat _A;
        A.convertTo(_A, CV_MAKETYPE(CV_32F, 1));
        B = _A.clone().t();
        cv::gemm(_A, B, 1, Mat(), 0, dst);
        imwrite("gemm_" + std::to_string(i) + "_original.bmp", dst);
        
        A = imread(imgpath[i], CV_LOAD_IMAGE_COLOR);

        cv::Canny(A, dst, 50.0, 50.0, 3, false);
        imwrite("canny_color_" + std::to_string(i) + "_false_original.bmp", dst);

        cv::Canny(A, dst, 50.0, 50.0, 3, true);
        imwrite("canny_color_" + std::to_string(i) + "_true_original.bmp", dst);

        for (int j = 0; j < sizeof(kernelSize) / sizeof(int); j++) {
            cv::boxFilter(A, dst, -1, Size(kernelSize[j], kernelSize[j]));
            imwrite("boxFilter" + std::to_string(kernelSize[j]) + "_" + std::to_string(i) + "_original.bmp", dst);
        }

        //GFX filter2D support next window size = 3,5,7,9,11
        for (int j = 0; j < 5; j++) {
            Mat kernel = Mat::eye(Size(kernelSize[j], kernelSize[j]), CV_32FC1);
            cv::filter2D(A, dst, -1, kernel);
            imwrite("filter2D" + std::to_string(kernelSize[j]) + "_" + std::to_string(i) + "_original.bmp", dst);
        }

        //GFX sepFilter2D support next window size = 3,5,7,9,11
        for (int j = 0; j < 5; j++) {
            Mat kernelX = Mat::eye(Size(kernelSize[j], 1), CV_32FC1) * 0.5f;
            Mat kernelY = Mat::eye(Size(kernelSize[j], 1), CV_32FC1) * 0.3f;
            cv::sepFilter2D(A, dst, -1, kernelX, kernelY);
            imwrite("sepFilter2D" + std::to_string(kernelSize[j]) + "_" + std::to_string(i) + "_original.bmp", dst);
        }

        for (int j = 0; j < 6; j++) {
            Mat kernel = Mat::ones(kernelSize[j], kernelSize[j], CV_8U);
            cv::erode(A, dst, kernel);
            imwrite("morph" + std::to_string(kernelSize[j]) + "_" + std::to_string(i) + "_original.bmp", dst);
        }
    }

    return 0;
}