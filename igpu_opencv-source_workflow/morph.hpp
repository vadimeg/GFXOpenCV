#include "igpu.hpp"


namespace cv {
namespace igpu {

#define X_MINMAX_CACHE_EFFECTIVE_SIZE_T(xCacheSizeT, cn, depth, pixRadius)  (X_CACHE_SIZE_T(xCacheSizeT, cn, depth) - 2 * cn * (pixRadius))
#define Y_MINMAX_CACHE_EFFECTIVE_SIZE_T(yCacheSizeT, cn, depth, pixRadius)  (Y_CACHE_SIZE_T(yCacheSizeT, cn, depth) - 2 * (pixRadius))

template< typename T, int cn, int depth, bool isMax, int kernelType, int xWindowSizePix, int yWindowSizePix, int yX, int yY, int xCacheSizeT, int yCacheSizeT >
__declspec(target(gfx_kernel))
void minmax_noBorders_tiled(const uchar* restrict srcptr, int srcStep,
                                uchar* restrict dstptr, int dstStep,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd) {
    cilk_for (int y = yItersStart; y <= yItersEnd; y++)
        cilk_for (int x = xItersStart; x <= xItersEnd; x++) {
            const int yRadius = yWindowSizePix / 2;
            const int xRadius = xWindowSizePix / 2;
            const int X_CACHE_EFFECTIVE_SIZE_T_ = X_MINMAX_CACHE_EFFECTIVE_SIZE_T(xCacheSizeT, cn, depth, xWindowSizePix / 2);
            const int Y_CACHE_EFFECTIVE_SIZE_T_ = Y_MINMAX_CACHE_EFFECTIVE_SIZE_T(yCacheSizeT, cn, depth, yWindowSizePix / 2);
            
            T cachedColRes[Y_CACHE_EFFECTIVE_SIZE_T_][X_CACHE_SIZE_T_];

            T cachedLeftRes[Y_CACHE_EFFECTIVE_SIZE_T_][X_CACHE_SIZE_T_];
            T cachedRightRes[Y_CACHE_EFFECTIVE_SIZE_T_][X_CACHE_SIZE_T_];

            const int LEFT = 0;
            const int RIGHT = 1;
            
            //for (int yChan = 1; yChan < Y_CACHE_SIZE_T_; yChan++) {
            //    cachedLeftRightRes[0][LEFT] = in[0][xChan];
            //    cachedLeftRightRes[Y_CACHE_SIZE_T_ - 1][RIGHT] = in[Y_CACHE_SIZE_T_ - 1][xChan];
            //}
            

            for (int yChan = 0; yChan < Y_CACHE_SIZE_T_; yChan++) {
                T cachedSrcRow[X_CACHE_SIZE_T_];
                T cachedSrcRowRev[X_CACHE_SIZE_T_];
                
                const uchar* src = srcptr + ((y * Y_CACHE_EFFECTIVE_SIZE_T_ + yChan) * srcStep + x * X_CACHE_EFFECTIVE_SIZE_T_) * depth;
                //__assume_aligned(src, 32);
                memcpy_(cachedSrcRow, 
                        src,
                        sizeof(cachedSrcRow));
                const int yChanRev = Y_CACHE_SIZE_T_ - yChan - 1;
                
                const uchar* srcr = srcptr + ((y * Y_CACHE_EFFECTIVE_SIZE_T_ + yChanRev) * srcStep + x * X_CACHE_EFFECTIVE_SIZE_T_) * depth;
                //__assume_aligned(srcr, 32);
                memcpy_(cachedSrcRowRev,
                        srcr,
                        sizeof(cachedSrcRowRev));
                
                if ((yChan % yWindowSizePix) == 0)
                    cachedLeftRes[yChan][:] = cachedSrcRow[:];
                else
                    cachedLeftRes[yChan][:] = min(cachedLeftRes[yChan - 1][:], cachedSrcRow[:]);
                //simd_for (int xChan = 0; xChan < X_CACHE_SIZE_T_; xChan++) 
                //    cachedLeftRightRes[yChan][xChan][LEFT] = ((yChan % yWindowSizePix) == 0) ?
                //        cachedSrcRow[xChan] : min(cachedLeftRightRes[yChan - 1][xChan][LEFT], cachedSrcRow[xChan]);
                
                if ((yChanRev % yWindowSizePix) == 0) {
                    cachedRightRes[yChanRev][:] = cachedSrcRowRev[:];
                }
                else {
                    cachedRightRes[yChanRev][:] = min(cachedRightRes[yChanRev + 1][:], cachedSrcRowRev[:]);
                }
                //simd_for (int xChan = 0; xChan < X_CACHE_SIZE_T_; xChan++) 
                //    cachedLeftRightRes[yChanRev][xChan][RIGHT] = ((yChanRev % yWindowSizePix) == 0) ?
                //        cachedSrcRowRev[xChan] : min(cachedLeftRightRes[yChanRev + 1][xChan][RIGHT], cachedSrcRowRev[xChan]);
                
            }
            
            for (int yChan = 0; yChan < Y_CACHE_EFFECTIVE_SIZE_T_; yChan++)
                cachedColRes[yChan][:] = min(cachedLeftRes[yChan + 2 * yRadius][:], cachedRightRes[yChan][:]);
            

            for (int yRes = 0; yRes < Y_CACHE_EFFECTIVE_SIZE_T_; yRes++) {
                T cache[X_CACHE_EFFECTIVE_SIZE_T_];
                simd_for (int xChan = cn * xRadius; xChan < cn * xRadius + X_CACHE_EFFECTIVE_SIZE_T_; xChan++) {
                    T res = cachedColRes[yRes][xChan - cn * xRadius];
                    for (int xRes = -xRadius + 1; xRes <= xRadius; xRes++)
                        res = isMax ?
                            max(res, cachedColRes[yRes][xChan + xRes * cn]) :
                            min(res, cachedColRes[yRes][xChan + xRes * cn]);
                    cache[xChan - cn * xRadius] = res;
                }

                //if (*(dstptr + yRes) == 11)
                memcpy_(dstptr + ((y * Y_CACHE_EFFECTIVE_SIZE_T_ + yRes + yRadius) * srcStep + x * X_CACHE_EFFECTIVE_SIZE_T_ + cn * xRadius) * depth,
                            cache,
                            sizeof(cache));
            }
        }
}

#include "def.h"
class MinMax {
public:
    MinMax(Mat& _src, Mat& _dst, const ImageOperationLayout& _layout, Size& _ksize,
           Mat _kernel, int _kernelType, bool _isMax, Point _anchor)
        :
        src(_src),
        dst(_dst),
        anchor(_anchor),
        layout(_layout),
        ksize(_ksize),
        kernel(_kernel),
        kernelType(_kernelType),
        isMax(_isMax) {
        
        srcptr = src.data;
        dstptr = dst.data;
        srcStep = src.step.p[0];
        dstStep = dst.step.p[0];
        cn = src.channels();
        depth = 1;
    }

    static Size elementsPerIter(const Mat& m, const Mat& kernel) {   
        const int r = kernel.cols / 2;
        const int d = kernel.cols / 2;
        const int xCacheSizeT = 64/* + m.channels() * 2 * r*/;                                                                                                    \
        const int yCacheSizeT_ = (((2 * 1024) / 64) - (2 * r + 2)) / 3;                                                                                                                \
        const int yCacheSizeT = yCacheSizeT_ > d ? yCacheSizeT_ : d;    

        return Size(X_MINMAX_CACHE_EFFECTIVE_SIZE_T(xCacheSizeT, m.channels(), m.depth(), kernel.cols / 2),
                    Y_MINMAX_CACHE_EFFECTIVE_SIZE_T(yCacheSizeT, m.channels(), m.depth(), kernel.rows / 2));
    }

    int onCpu_noBorders_tiled() {
        const Rect& iters = layout.gpuItersRect();
    
        {
#define inputCase(dArg, xCacheSize) {                                                                                                                   \
    const int d = dArg;                                                                                                                     \
    const int r = d / 2;                                                                                                                  \
    const int xCacheSizeT = xCacheSize/* + cn * 2 * r*/;                                                                                                    \
    const int yCacheSizeT_ = (((2 * 1024) / xCacheSize) - (2 * r + 2)) / 3;                                                                                                                \
    const int yCacheSizeT = yCacheSizeT_ > d ? yCacheSizeT_ : d;                                                                                                                          \
    if (isMax)                                                                                                                                                \
        minmax_noBorders_tiled<uchar, cn, 1, true, CV_SHAPE_RECT, d, d, r, r, xCacheSizeT, yCacheSizeT >(srcptr, srcStep,               \
                        dstptr, dstStep,                                                                                                                      \
                        iters.x, iters.x + iters.width - 1, iters.y, iters.y + iters.height - 1);                                                                                                                               \
    else                                                                                                                                                      \
        minmax_noBorders_tiled<uchar, cn, 1, false, CV_SHAPE_RECT, d, d, r, r, xCacheSizeT, yCacheSizeT >(srcptr, srcStep,              \
                        dstptr, dstStep,                                                                                                                      \
                        iters.x, iters.x + iters.width - 1, iters.y, iters.y + iters.height - 1);                                                             \
    return -1;                                                                                                                             \
}   

            const int cn = 3;
            const int depth = 1;
            switch(ksize.width) {
                case 3: inputCase(3, 64);
                //case 5: inputCase(5, 128);
                //case 7: inputCase(7, 128);
                //case 9: inputCase(9, 128);
                default: 
                    onGpu_end(-1);
            }

        }
        return -1;

    }

    int onGpu_noBorders_tiled() {
        const Rect& iters = layout.gpuItersRect();
    
        {
	        _GFX_share((void*)src.datastart, (size_t)(src.dataend - src.datastart));
	        _GFX_share((void*)dst.datastart, (size_t)(dst.dataend - dst.datastart));

#define inputCase(dArg, xCacheSize) {                                                                                                                   \
    const int d = dArg;                                                                                                                     \
    const int r = d / 2;                                                                                                                  \
    const int xCacheSizeT = xCacheSize/* + cn * 2 * r*/;                                                                                                    \
    const int yCacheSizeT_ = (((2 * 1024) / xCacheSize) - (2 * r + 2)) / 3;                                                                                                                \
    const int yCacheSizeT = yCacheSizeT_ > d ? yCacheSizeT_ : d;                                                                                                                          \
    if (isMax)                                                                                                                                                \
        return _GFX_offload(&minmax_noBorders_tiled<uchar, cn, 1, true, CV_SHAPE_RECT, d, d, r, r, xCacheSizeT, yCacheSizeT >, srcptr, srcStep,               \
                        dstptr, dstStep,                                                                                                                      \
                        iters.x, iters.x + iters.width - 1, iters.y, iters.y + iters.height - 1);                                                                                                                               \
    else                                                                                                                                                      \
        return _GFX_offload(&minmax_noBorders_tiled<uchar, cn, 1, false, CV_SHAPE_RECT, d, d, r, r, xCacheSizeT, yCacheSizeT >, srcptr, srcStep,              \
                        dstptr, dstStep,                                                                                                                      \
                        iters.x, iters.x + iters.width - 1, iters.y, iters.y + iters.height - 1);                                                                                                                                \
}   

            const int cn = 3;
            const int depth = 1;

            switch(ksize.width) {
                case 3: inputCase(3, 64);
                //case 5: inputCase(5, 128);
                //case 7: inputCase(7, 128);
                //case 9: inputCase(9, 128);
                default: 
                    onGpu_end(-1);
            }

        }
        return -1;

    }
    
    void onGpu_end(int k) {
        if (k >= 0)
            _GFX_wait(k);
	    _GFX_unshare((void*)src.datastart);
	    _GFX_unshare((void*)dst.datastart);
    }

    void onCpu_atGpuBorder(int borderType, const Scalar& borderValue) {
        Rect rois[4];
        computeCpuBorderRois(rois, layout);

        if (!isMax) {
            for (auto& roi : rois)
                if (roi.width && roi.height)
                    cv::erode(src(roi), dst(roi), kernel, Point(-1, -1), 1, borderType, borderValue);
        } else {
            for (auto& roi : rois)
                if (roi.width && roi.height)
                    cv::dilate(src(roi), dst(roi), kernel, Point(-1, -1), 1, borderType, borderValue);
        }
    }
    
private:


    int cn;
    int depth;

    uchar* srcptr;
    int srcStep;
    uchar* dstptr;
    int dstStep;

    const ImageOperationLayout& layout;

    const Point anchor;

    const int kernelType;
    const bool isMax;

    Size ksize;
    
    Mat& kernel;
    Mat& src;
    Mat& dst;
};

bool morphologyEx( InputArray _src, OutputArray _dst, int op,
                       InputArray _kernel, Point anchor, int iterations,
                       int borderType, const Scalar& borderValue ) {
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    Mat kernel = _kernel.getMat();
    Size ksize = !kernel.empty() ? kernel.size() : Size(3, 3), ssize = _src.size();

    if (kernel.cols != kernel.rows)
        return false;

    const int d = kernel.cols;                                                                                                         
    const int r = d / 2; 

    if (iterations == 0)
        return false;
    
    if (anchor != Point(r, r))
        return false;

    if (src.total() < 1 / d || src.cols < d * 1 || src.rows < d * 1 || d % 2 != 1 || d > 9 || d < 3)
        return false;

    if (kernel.empty()) {
        kernel = getStructuringElement(MORPH_RECT, Size(1 + iterations * 2, 1 + iterations * 2));
        anchor = Point(iterations, iterations);
        iterations = 1;
    } else if( iterations > 1 && countNonZero(kernel) == kernel.rows * kernel.cols ) {
        anchor = Point(anchor.x*iterations, anchor.y*iterations);
        kernel = getStructuringElement(MORPH_RECT,
                                       Size(ksize.width + (iterations - 1) * (ksize.width - 1),
                                            ksize.height + (iterations - 1) * (ksize.height - 1)),
                                       anchor);
        iterations = 1;
    }

    //Point anchor = anchor0;//normalizeAnchor(anchor0, ksize);


    float gpuPart = 1.0f;//0.9f;

    tstart
                                                                                                          
    if (src.type() != CV_8UC3)  
        return false;                 

    const int cn = 3;  
    const int depth = 1;
    ImageOperationLayout layout;

    gpuPart = layout.accept(ImageSize(cn * src.cols, src.rows), gpuPart,
                    AtomSizeBytes(src.elemSize(), 1), AtomSizeBytes(depth, 1),
                    ItersStart(cn * r, r), MinMax::elementsPerIter(src, kernel),
                    LeftTopBorderSize(cn * r, r), RightBotBorderSize(cn * r, r));
    cout << "gpuPart " << gpuPart << endl;
    if (gpuPart <= 0.f)
        return false;

    MinMax filter(src, dst, layout, ksize, kernel, CV_SHAPE_RECT, op == MORPH_DILATE, anchor);
    std::cout << "iters " << layout.gpuItersRect() << std::endl;
    std::cout << "per iter " << MinMax::elementsPerIter(src, kernel) << std::endl;
t2start
    filter.onCpu_atGpuBorder(borderType, borderValue);
t2end("borders ")
    int k = filter.onGpu_noBorders_tiled();
    if (k <= 0) {
        cout << "k <= 0" << endl;
        return false;
    }
t2start
    filter.onGpu_end(k);    
t2end("wait ")           

    tend("ALL TIME                                   ")

    
t2start
    
    filter.onCpu_noBorders_tiled();
    //if (layout.mainCpuWorkRect().width && layout.mainCpuWorkRect().height)
    //    cv::morphologyEx(src(layout.mainCpuWorkRect()), dst(layout.mainCpuWorkRect()), op, kernel, anchor, iterations, borderType, borderValue);
t2end("cpu                          ")

    return true;
}

}
}
