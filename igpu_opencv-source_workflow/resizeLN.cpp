#include "igpu_comp_primitives.hpp"
#include "cv_igpu_interface.hpp"

#include <opencv2\imgproc.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <gfx\gfx_rt.h>
#include <cilk\cilk.h>

#include <iostream>
#include <sstream>
#include <chrono>
#include <ratio>

using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace igpu;

#define cilk_for for
#define unrolled_for for

#define loadPixFromBuf(cacheBuffer, pixBuffer, bufferedX, bufferedY, cn) { \
    for (int pixCounter = 0; pixCounter < cn; pixCounter++) \
        pixBuffer[j] = cacheBuffer[bufferedY][bufferedX * cn + pixCounter]; \
} \

template <typename T, typename WT, int cn, int xCacheSize, int yCacheSize>
void resizeLinear_noBorders_tiled(const uchar* restrict srcptr, int srcStep, uchar* restrict dstptr, int dstStep,
                                  float ifx, float ify, int xTilePixSize, int yTilePixSize,
                                  int leftItersBorder, int rightItersBorder, int topItersBorder, int bottomItersBorder) {
    cilk_for(int x = leftItersBorder; x < rightItersBorder; x++) {
        cilk_for(int y = topItersBorder; y < bottomItersBorder; y++) {
            const int xTileTSize = xTilePixSize * cn;
            const int yTileTSize = yTilePixSize;
            /* Cache size for storaging data on every ...Размер кеша для считывания по каждой оси (для одного треда) */
            /* Cache size calculating without round handling */
            const int xSrcTileCache = xTilePixSize / 2 * ifx; // use gfx_floor() before this kernel
            const int ySrcTileCache = yTilePixSize / 2 * ify;
            const int xCacheTSize = xTilePixSize * cn;

            T cachedSrc[yCacheSize][xCacheSize];
            for (int i = 0; i < yCacheSize; i++)
                memcpy(&cachedSrc[i][0], srcptr + (y * ySrcTileCache * srcStep + x * xSrcTileCache) * cn, xCacheTSize * sizeof(T));


            for (int i = 0; i < yTilePixSize; i++) {
                for (int j = 0; j < xTilePixSize; j++) {
                    int dx = x * xTilePixSize + j; //OK
                    int dy = y * yTilePixSize + i * yTilePixSize;
                    float sx = ((dx + 0.5f) * ifx - 0.5f), sy = ((dy + 0.5f) * ify - 0.5f);
                    int x = gfx_floor(sx); /* C++ cast works like trunc() and floor() for positive numbers */
                    int y = gfx_floor(sy);
                    float u = sx - x;
                    float v = sy - y;

                    if (x < 0) x = 0, u = 0;
                    //if (x >= src_cols) x = src_cols - 1, u = 0;
                    if (y < 0) y = 0, v = 0;
                    //if (y >= src_rows) y = src_rows - 1, v = 0;   

                    int y_ = y + 1;
                    int x_ = x + 1;

                    float u1 = 1.f - u;
                    float v1 = 1.f - v;
                    WT data0[cn];
                    WT data1[cn];
                    WT data2[cn];
                    WT data3[cn];

                    loadPixFromBuf(cachedSrc, data0, j, i, cn)
                    loadPixFromBuf(cachedSrc, data0, j, i, cn)
                    loadPixFromBuf(cachedSrc, data0, j, i, cn)
                    loadPixFromBuf(cachedSrc, data0, j, i, cn)

                    T uval[cn];
                    for (int i = 0; i < cn; i++)
                        uval[i] = u1 * v1 * data0[i] + u * v1 * data1[i] + u1 * v * data2[i] + u * v * data3[i];
                }
            }
        }
    }
}

struct ResizeLinearGPUTask {
    bool async;
    
    Offload2dBuffer src;
    Offload2dBuffer dst;
    
    float ifx;
    float ify;
    
    using ItersRectsT = ImageOperationLayout::ArrayT;
    ImageOperationLayout::ArrayT itersRects;
};

class GFXResizeLinear {
public:
    int start(const ResizeLinearGPUTask& _task) { // TODO: Repair work with pointer to _task (add copy constructor)
        task = &_task;

        _GFX_share(task->src.memoryStart, task->src.wholeSizeBytes);
        _GFX_share(task->dst.memoryStart, task->dst.wholeSizeBytes);
        kernelStart<3>();
        
        return 1;
    }

    void finalize(int k) {
        if (k > 0) { 
            cout << "f k = " << k << endl;
            _GFX_wait(k);
            cout << "f" << endl;
            _GFX_unshare(task->src.memoryStart);
            _GFX_unshare(task->dst.memoryStart);
        }
    }
    
private:
    template<int cn>
    int kernelStart() {
        const int yTiles = 4; // Experimantal!
        const int xTilePixSize = 32;
        const int yTilePixSize = yTiles * 2;

        /* Количество считываемых байт из src для обработки одного тайла dst по x */
        const int xCacheTSize = cn * xTilePixSize;
        const int yCacheTSize = yTilePixSize;

        int lastK = -1;
        for (const auto& iters : task->itersRects) { // TODO: Design mistake! Waiting only last task!
            int k = _GFX_offload(&resizeLinear_noBorders_tiled<uchar, int, cn, xCacheTSize, yCacheTSize>,
                                    (const uchar*)task->src.buffer, task->src.step, (uchar*)task->dst.buffer, task->dst.step, 
                                    task->ifx, task->ify, xTilePixSize, yTilePixSize,
                                    iters.x, iters.x + iters.width - 1, iters.y, iters.y + iters.height - 1);
            cout << "k = " << k << endl;

            if (k < 0) {
                std::cout << "k < 0" << std::endl;
                finalize(lastK);
                return -1;
            }

            lastK = k;
        }
        
        return lastK;
    }

    const ResizeLinearGPUTask* task;
};

class ResizeLinearAdviser : public Adviser {
public:
    bool accept(InputArray _src, OutputArray _dst, Size size, double _ifx, double _ify, int interpolation) {
        Mat src = _src.getMat();
        Mat dst = _dst.getMat();
        /* Image size check: if image size smaller then minimal (minimal size with best performance) -
         * return */
        /* Cutting dst image on tiles and prepare kernel task data */
        const int yTiles = 4; // Experimantal!
        const int xTilePixSize = 32;
        const int yTilePixSize = yTiles * 2;
        const int cn = CV_MAT_CN(src.type());

        /* Количество считываемых байт из src для обработки одного тайла dst по x */
        int xCacheTSize = cn * xTilePixSize; 
        int yCacheTSize = yTilePixSize;

        /* Cropping image on parts: CPU Border, GPU, CPU */
        float gpuPart = 0.7f; // !!!!! MISTAKE
        gpuPart = layout.accept(ImageSize(src.cols, src.rows), gpuPart,
                                AtomSizeBytes(src.elemSize(), 1), AtomSizeBytes(src.elemSize(), 1),
                                ItersStart(0, 0), AtomsPerIteration(xTilePixSize, yTilePixSize),
                                LeftTopBorderSize(0, 0), RightBotBorderSize(1, 0)); // TODO: Recheck this!!
        if (gpuPart < 0.f)
            return false;

        task.async = false;
        task.ifx = _ifx;
        task.ify = _ify;

        const int buffersNum = 2;
        Offload2dBuffer* buffers[buffersNum] = {&task.src, &task.dst};
        Mat* mats[buffersNum] = {&src, &dst};

        for (int i = 0; i < buffersNum; i++) {
            buffers[i]->buffer = mats[i]->data;
            buffers[i]->step = mats[i]->step.p[0];
            buffers[i]->memoryStart = (void*) mats[i]->datastart;
            buffers[i]->wholeSizeBytes = (size_t)(mats[i]->dataend - mats[i]->datastart);
        }
            
        task.itersRects = layout.primeLayout().workRects();
        return true;
    }

    using GpuTaskT = ResizeLinearGPUTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }

private:
    ResizeLinearGPUTask task;
};

bool resizeLinear(InputArray _src, OutputArray _dst, double ifx, double ify) {
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    /* Check image type with supported types */
    if (src.type() != CV_8UC3)
        return false;

    return performHeteroTask<ResizeLinearAdviser, GFXResizeLinear, 
                             void(*)(InputArray _src, OutputArray _dst, Size dsize, double inv_scale_x, double inv_scale_y, int interpolation), Mat>
                             (ResizeLinearAdviser(), GFXResizeLinear(), &cv::resize, src, dst, Size(), ifx, ify, INTER_LINEAR);
}

__declspec(target(gfx))
inline int mad24(int a, int b, int c) { return a * b + c; }

__declspec(target(gfx))
inline int gfx_min(int a, int b) { return (a < b) ? (a) : (b); }

__declspec(target(gfx))
inline int gfx_floor(float value) { return (value >= 0) ? (value) : (value - 1); }

/*template <typename WT, typename T>
__declspec(target(gfx))
inline void loadPixFromBuf(T *readBuffer, WT *data, int bufferedX, int cn, int offset = 0) {
    for (int j = 0; j < cn; j++)
        data[j] = readBuffer[bufferedX * cn + j + offset];
}*/


void measuredTime(Mat &A, Mat &B, int parameter, int width) {
    std::chrono::high_resolution_clock time;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point finish;
    std::chrono::high_resolution_clock::time_point result;
    duration<double> mintime;

    for (int i = 0; i < 10; i++) {
        start = std::chrono::high_resolution_clock::now();
        resize(A, B, cv::Size(width, width), 0, 0, parameter);
        finish = std::chrono::high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(finish - start);

        if (!i)
            mintime = time_span;
        else {
            if (time_span.count() < mintime.count())
                mintime = time_span;
        }
    }

    /*std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(finish - start);*/
    std::cout << "Time " << mintime.count() << " seconds." << std::endl;
}

int main() {
    Mat source;
    UMat _A;
    Mat A;
    Mat B;
    Mat C;
    Mat D;
    Mat E;
    A = imread("C:\\Resize\\1.jpg", CV_LOAD_IMAGE_COLOR);
    Mat dst[] = {A, B, C, D, E};
    String names[] {"b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"};
    int parameters[] = {INTER_LINEAR, INTER_NEAREST, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4};
    int width_arr[] = {500, 1000, 1500, 2000, 5000, 10000, 16000};

    std::cout << "CPU Resize" << std::endl;
    for (int i = 0; i < sizeof(parameters) / sizeof(int); i++) {
        std::cout << "Current dst image width = " << 5000 << std::endl;
        measuredTime(A, dst[i], INTER_LINEAR, 5000);
        //imwrite(names[i], dst[i]);
    }

    int type = A.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    int wdepth = std::max(depth, CV_32S), wtype = CV_MAKETYPE(wdepth, cn);

    int width = 512;
    int height = 512;
    double inv_scale_x = (double)width / A.cols;
    double inv_scale_y = (double)height / A.rows;
    B.create(Size(width, height), A.type());

    double inv_fx = 1.0 / inv_scale_x, inv_fy = 1.0 / inv_scale_y;
    /*if (depth > CV_32S) {
    resizeLN<uchar, double, 1>(A.ptr<uchar>(), A.step.buf[0], A.rows, A.cols, B.ptr<uchar>(), B.step.buf[0], B.rows, B.cols, inv_fx, inv_fy);
    } else*/
    std::cout << "\nGFX Resize\n";
#ifdef OFFLOAD
    _GFX_share((void*)A.datastart, A.dataend - A.datastart);
    _GFX_share((void*)B.datastart, B.dataend - B.datastart);
#endif
    std::chrono::high_resolution_clock time;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point finish;
    std::chrono::high_resolution_clock::time_point result;
    duration<double> mintime;
    for (int i = 0; i < 1; i++) {
        start = std::chrono::high_resolution_clock::now();
/*#ifdef OFFLOAD
        _GFX_offload(resizeLN<uchar, int, 3>, A.ptr<uchar>(), A.step.buf[0], A.rows, A.cols, B.ptr<uchar>(), B.step.buf[0], B.rows, B.cols, inv_fx, inv_fy);
        _GFX_wait();
#else
        //resizeLN<uchar, int, 3>(A.ptr<uchar>(), A.step.buf[0], A.rows, A.cols, B.ptr<uchar>(), B.step.buf[0], B.rows, B.cols, inv_fx, inv_fy);
#endif*/
        resizeLinear(A, B, inv_fx, inv_fy);
        finish = std::chrono::high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(finish - start);

        if (true)
            mintime = time_span;
        else {
            if (time_span.count() < mintime.count())
                mintime = time_span;
        }
    }

    std::cout << "Time " << mintime.count() << " seconds." << std::endl;
    imwrite("Result.jpg", B);
    getchar();
    return 1;
}