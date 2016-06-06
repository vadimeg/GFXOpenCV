#include <mathimf.h>
#include "cv_igpu_interface.hpp"
#include "igpu_comp_primitives.hpp"
#include "filter2d.hpp"


namespace cv {
namespace igpu {

#define linearFilter_atPixelChannal(x, y, loadSrc, storeDst) {                        \
        SUM_T sum = minSum;                                                           \
        for (int j = 0; j < yWindowSizePix; j++)                                        \
            simd_for (int i = 0; i < xWindowSizePix; i++) {                                  \
                T chan;                                                               \
                loadSrc(x + cn * i - xRadius * cn, y + j - yRadius, chan);             \
                sum += (SUM_T) (chan * kernel[j * xWindowSizePix + i]);                  \
            }                                                                         \
        const T chan = (T)clamp(sum + delta, minSum, maxSum);                          \
        storeDst(x, y, chan);                                                       \
}

template< typename T, typename KER_T, typename SUM_T, SUM_T minSum, SUM_T maxSum, int cn, int kernelSize, int xTileTSize, int yTileTSize, int yIters >
__declspec(target(gfx_kernel))
void linearFilter_noBorders_tiled(const uchar* restrict srcptr, int srcStep,
                                uchar* restrict dstptr, int dstStep,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd,
                                SUM_T delta, const KER_T* restrict kernel_) {
    
    cilk_for (int y_ = yItersStart; y_ <= yItersEnd; y_++)
        cilk_for (int x = xItersStart; x <= xItersEnd; x++) {
            const int depth = sizeof(T);

            const int xWindowSizePix = kernelSize;
            const int yWindowSizePix = kernelSize;
            const int xRadius = xWindowSizePix / 2;
            const int yRadius = yWindowSizePix / 2;

            const int xCacheTSize = XCacheSize<xTileTSize, cn * xRadius>::value;
            const int yCacheTSize = YCacheSize<yTileTSize, yRadius>::value;

            KER_T kernel[xWindowSizePix * yWindowSizePix];
            __assume_aligned(kernel_, 16);

            kernel[0 : xWindowSizePix * yWindowSizePix] = kernel_[0 : xWindowSizePix * yWindowSizePix];
            
            const int y = yIters * y_;

            atTile_initCache();
            atTile_loadCache(x, y);
            
            for (int yLocalChan = yRadius; yLocalChan < yRadius + yTileTSize; yLocalChan++) {
                T cachedDstRow[xTileTSize];
                const int dstXCacheOffset = xRadius * cn;
                simd_for (int xLocalChan = xRadius * cn; xLocalChan < xRadius * cn + xTileTSize; xLocalChan++)
                    linearFilter_atPixelChannal(xLocalChan, yLocalChan, loadSrc_tiled, storeDst_rowCached);

                memcpy_(dstptr + ((y * yTileTSize + yLocalChan) * dstStep + x * xTileTSize + dstXCacheOffset) * depth,
                        cachedDstRow,
                        sizeof(cachedDstRow));
            }

        }
}


struct Filter2dGpuTask {
    Filter2dGpuTask() {}
    Filter2dGpuTask(bool _async, 
                    const Offload2dBuffer& _src,
                    const Offload2dBuffer& _dst,
                    const Offload2dBuffer& _kernel,
                    const ImageOperationLayout::ArrayT& _itersRects,
                    int _windowSize,
                    float _delta) :
                    async(_async),
                    src(_src),
                    dst(_dst),
                    kernel(_kernel),
                    windowSize(_windowSize),
                    delta(_delta),
                    itersRects(_itersRects) {}

    bool async;

    Offload2dBuffer src;
    Offload2dBuffer dst;
    //int ddepth;
    Offload2dBuffer kernel;

    using ItersRectsT = ImageOperationLayout::ArrayT;
    ImageOperationLayout::ArrayT itersRects;

    int windowSize;
    float delta;
};

class Filter2dAdviser : public Adviser {
public:
    Filter2dAdviser() {
    }

    bool accept(InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernel, Point anchor0,
                   double delta, int borderType) {
        const int d = _kernel.rows();                                                                                                         
        const int r = d / 2;

        if (anchor0 != Point(r, r))
            return false;

        if (_src.total() < 1 / d || _src.cols() < d * 1 || _src.rows() < d * 1 || d % 2 != 1)
            return false;
        if (!inRange(d, 3, 11))
            return false;

        if (_src.type() != CV_8UC3)  
            return false;
                                     
        const int cn = 3;  
        const int depth = 1;

        float gpuPart = 0.8f;

        Mat src = _src.getMat();
        Mat dst = _dst.getMat();
        Mat kernel = _kernel.getMat();
        
        const int xTileTSize = 128;
        const int xCacheTSize = xCacheSize(xTileTSize, cn * r);
        const int yTileTSize_ = (igpuRegisterFileSize - xTileTSize * (1 + sizeof(float)) - sizeof(float) * d * d) / xCacheTSize;
        const int yTileTSize = (yTileTSize_ > d) ? yTileTSize_ : d;

        gpuPart = layout.accept(ImageSize(cn * src.cols, src.rows), gpuPart,
                        AtomSizeBytes(src.elemSize(), 1), AtomSizeBytes(depth, 1),
                        ItersStart(cn * r, r), AtomsPerIteration(xTileTSize, yTileTSize),
                        LeftTopBorderSize(cn * r, r), RightBotBorderSize(cn * r, r));
        cout << "gpuPart " << gpuPart << endl;
        if (gpuPart < 0.f) {
            return false;
        }

        task.async = false;
        task.delta = (float) delta;
        task.windowSize = kernel.cols;

        const int buffersNum = 3;
        Offload2dBuffer* buffers[buffersNum] = {&task.src, &task.dst, &task.kernel};
        Mat* mats[buffersNum] = {&src, &dst, &kernel};

        for (int i = 0; i < buffersNum; i++) {
            buffers[i]->buffer = mats[i]->data;
            buffers[i]->step = mats[i]->step.p[0];
            buffers[i]->memoryStart = (void*) mats[i]->datastart;
            buffers[i]->wholeSizeBytes = (size_t)(mats[i]->dataend - mats[i]->datastart);
        }
            
        task.itersRects = layout.primeLayout().workRects();

        return true;
    }

    using GpuTaskT = Filter2dGpuTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }
    
    //using CpuRoisT = ImageOperationLayoutCpu::ArrayT;
    //const GpuTaskT& advisedCpu() const {
    //    return task;
    //}
    //
    //using BorderRoisT = ImageBorderOperationLayout::ArrayT;
    //const BorderRoisT& advisedBorder() const {
    //    return task;
    //}

private:
    Filter2dGpuTask task;
};

const int yFilter2dTiles = 1;
#include "def.h"
class Filter2d {
public:
    Filter2d() {}

    int start(const Filter2dGpuTask& _task) {
        task = &_task;

        if (!inRange(task->windowSize, 3, 11))
            return -1;

	    _GFX_share(task->src.memoryStart, task->src.wholeSizeBytes);
	    _GFX_share(task->dst.memoryStart, task->dst.wholeSizeBytes);
	    _GFX_share(task->kernel.memoryStart, task->kernel.wholeSizeBytes);
        
        const int cn = 3;

        switch(task->windowSize) {
            case 3: return inputCase<3, 128, cn>();
            case 5: return inputCase<5, 128, cn>();
            case 7: return inputCase<7, 128, cn>();
            case 9: return inputCase<9, 128, cn>();
            case 11: return inputCase<11, 128, cn>();
        }

        return -1;
    }

    void finalize(int k) {
        if (k > 0) { 
            _GFX_wait(k);
	        _GFX_unshare(task->src.memoryStart);
	        _GFX_unshare(task->dst.memoryStart);
	        _GFX_unshare(task->kernel.memoryStart);
        }
    }
    
private:

    template<int d, int xTileTSize, int cn>
    int inputCase() {
        const int r = d / 2;
        const int xCacheTSize = XCacheSize<xTileTSize, cn * r>::value;
        const int yTileTSize_ = (igpuRegisterFileSize - xTileTSize * (1 + sizeof(float)) - sizeof(float) * d * d) / xCacheTSize;
        const int yTileTSize = static_max(yTileTSize_, d);

        int lastK = -1;
        for (const auto& iters : task->itersRects) {
            int k = _GFX_offload(&linearFilter_noBorders_tiled<uchar, float, float, 0.f, 255.f/***/, cn, d, xTileTSize, yTileTSize, yFilter2dTiles>,
                                    (const uchar*)task->src.buffer, task->src.step,
                                    (uchar*)task->dst.buffer, task->dst.step,
                                    iters.x, iters.x + iters.width - 1, iters.y, iters.y + iters.height - 1,
                                    task->delta, (const float*)task->kernel.buffer);
            if (k < 0) {
                std::cout << "k < 0" << std::endl;
                finalize(lastK);
                return -1;
            }

            lastK = k;
        }

        return lastK;
    }

    const Filter2dGpuTask* task;
};



bool filter2D( InputArray _src, OutputArray _dst, int ddepth,
               InputArray _kernel, Point anchor0,
               double delta, int borderType ) {
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    Mat kernel = _kernel.getMat();
    
    if( ddepth < 0 )
        ddepth = src.depth();
 
    if (!makeMatSquare(kernel))
        return false;
 
    const int d = kernel.rows;                                                                                                         
    const int r = d / 2;      
 
    Point anchor = anchor0;//normalizeAnchor(anchor0, kernel.size());
 
    return performHeteroTask(Filter2dAdviser(), Filter2d(), &cv::filter2D, src, dst, ddepth, _kernel, anchor0, delta, borderType);
}


}
}
