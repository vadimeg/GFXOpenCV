
namespace cv {
namespace igpu {

#define bilateralFilter_atPixel(x, y, loadSrc, storeDst) {                                           \
        SUM_T sum[cn];                                                                             \
        SUM_T wsum = (SUM_T)0;                                                                     \
        sum[:] = (SUM_T)0;                                                                     \
        SUM_T val0[cn];                                                                                \
        for (int c = 0; c < cn; c++)                                                               \
            loadSrc(cn * x + c, y, val0[c]);                                                       \
        for (int k = 0; k < maxK; k++) {                                                           \
            float diff = 0.f;                                                                      \
            SUM_T val[cn];                                                                         \
            for (int c = 0; c < cn; c++) {                                                         \
                loadSrc((x + (int)space_ofs[2 * k + 1]) * cn + c, y + (int)space_ofs[2 * k], val[c]);         \
                diff += (float)abs(val[c] - val0[c]);                                              \
            }                                                                                      \
            float w = space_weight[k] * expf(diff * diff * gauss_color_coeff);             \
            wsum += w;                                                                             \
            sum[:] += val[:] * w;                                                               \
        }                                                                                          \
        for (int c = 0; c < cn; c++) {                                                               \
            const T chan = (T)(sum[c] / wsum);                                                    \
            storeDst(cn * x + c, y, chan);                                                      \
        }                                                                                           \
}



template< typename T, typename SUM_T, int cn, int depth >
__declspec(target(gfx_kernel))
void bilateral_noBorders(const uchar* restrict srcptr, int srcStep,
                                uchar* restrict dstptr, int dstStep,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd,
                                float gauss_color_coeff, int maxK, const float* restrict space_weight, const char* restrict space_ofs) {
    
    cilk_for (int y = yItersStart; y <= yItersEnd; y++)
        #pragma simd
        cilk_for (int x = xItersStart; x <= xItersEnd; x++)
            bilateralFilter_atPixel(x, y, loadSrc_direct_greedy, storeDst_direct);
}

template< typename T, typename SUM_T, int cn, int depth, int windowSize, int xCacheSizeT, int yCacheSizeT, int yIters >
__declspec(target(gfx_kernel))
void bilateral_noBorders_tiled(const uchar* restrict srcptr, int srcStep,
                                uchar* restrict dstptr, int dstStep,
                                int leftItersBorder, int rightItersBorder, int topItersBorder, int bottomItersBorder,
                                float gauss_color_coeff, int maxK, const float* restrict _space_weight, const char* restrict _space_ofs) {

    cilk_for (int y_ = topItersBorder; y_ <= bottomItersBorder; y_++)
        cilk_for (int x = leftItersBorder; x <= rightItersBorder; x++) {
            const int xWindowSizePix = windowSize;
            const int yWindowSizePix = windowSize;
            const int xRadius = xWindowSizePix / 2;
            const int yRadius = yWindowSizePix / 2;
            const int X_CACHE_EFFECTIVE_SIZE_T_ = X_CACHE_EFFECTIVE_SIZE_T(xCacheSizeT, cn, depth, xRadius);
            const int Y_CACHE_EFFECTIVE_SIZE_T_ = Y_CACHE_EFFECTIVE_SIZE_T(yCacheSizeT, cn, depth, yRadius);
            char space_ofs[2 * ((xWindowSizePix * yWindowSizePix * 3.15) / 4 + 1)];
            float space_weight[(xWindowSizePix * yWindowSizePix * 3.15) / 4 + 1];
            //memcpy(space_ofs, _space_ofs, 2 * maxK);
            //memcpy(space_weight, _space_weight, sizeof(float) * maxK);
            space_ofs[0 : 2 * maxK] = _space_ofs[0 : 2 * maxK];
            space_weight[0 : maxK] = _space_weight[0 : maxK];
            
            const int y = yIters * y_;

            atTile_initCache();
            atTile_loadCache(x, y);

            for (int yLocalPix = yRadius; yLocalPix < yRadius +  Y_CACHE_EFFECTIVE_SIZE_PIX(yCacheSizeT, cn, depth, yRadius); yLocalPix++)
                #pragma simd
                for (int xLocalPix = xRadius; xLocalPix < xRadius +  X_CACHE_EFFECTIVE_SIZE_PIX(xCacheSizeT, cn, depth, xRadius); xLocalPix++)
                    bilateralFilter_atPixel(xLocalPix, yLocalPix, loadSrc_tiled, storeDst_direct_tiled);

            for (int yIter = 1; yIter < yIters; yIter++) {
                cachedSrc[0 : yRadius * 2][:] = cachedSrc[Y_CACHE_SIZE_T_ - yRadius * 2 : yRadius * 2][:];
                const int y = y_ * yIters + yIter;
                atTilePart_loadCache(x, y, yRadius * 2, Y_CACHE_SIZE_T_ - 1);
                for (int yLocalPix = yRadius; yLocalPix < yRadius +  Y_CACHE_EFFECTIVE_SIZE_PIX(yCacheSizeT, cn, depth, yRadius); yLocalPix++)
                    #pragma simd
                    for (int xLocalPix = xRadius; xLocalPix < xRadius +  X_CACHE_EFFECTIVE_SIZE_PIX(xCacheSizeT, cn, depth, xRadius); xLocalPix++)
                        bilateralFilter_atPixel(xLocalPix, yLocalPix, loadSrc_tiled, storeDst_direct_tiled);
            }
        }
}


const int yBilatTiles = 4;
class Bilateral {
public:
    Bilateral(Mat& _src, Mat& _dst, const ImageOperationLayout& _layout, int _d, float _sigmaColor, float _sigmaSpace)
        : 
        src(_src),
        dst(_dst),
        sigmaColor(_sigmaColor),
        sigmaSpace(_sigmaSpace),
        windowSize(_d),
        layout(_layout),
        rows(_src.rows), 
        cols(_src.cols) {

        srcptr = src.data;
        dstptr = dst.data;
        srcStep = src.step.p[0];
        dstStep = dst.step.p[0];
        cn = src.channels();
        depth = 1;

        maxK = 0;
        gauss_color_coeff = -0.5f / (sigmaColor * sigmaColor);
        float gauss_space_coeff = -0.5f / (sigmaSpace * sigmaSpace);
        const int r = windowSize / 2;
        for( int i = -r; i <= r; i++ )
            for( int j = -r; j <= r; j++ ) {
                float rr = std::sqrt((float)i * i + (float)j * j);
                if ( rr > r )
                    continue;
                space_weight[maxK] = (float)std::expf(rr * rr * gauss_space_coeff);
                space_ofs[2 * maxK] = i;
                space_ofs[2 * maxK++ + 1] = j;
            }
    }

    int onGpu_noBorders_tiled() {
        const Rect& iters = layout.gpuItersRect();
    
        {
	        d_GFX_share(srcptr, getWholeMatSizeBytes(src));
	        d_GFX_share(dstptr, getWholeMatSizeBytes(dst));
	        d_GFX_share(space_weight, maxK * sizeof(float));
	        d_GFX_share(space_ofs, maxK * 2 * sizeof(int));

#define inputCase(dArg, xCacheSize) {                                                                                                                   \
    const int d = dArg;                                                                                                                     \
    const int r = d / 2;                                                                                                                  \
    const int xCacheSizeT = cn * (xCacheSize) + cn * 2 * r;                                                                                         \
    const int yCacheSizeT = ((2 * 1024 - d * d * 6) / xCacheSizeT > d) ? (2 * 1024 - d * d * 6) / xCacheSizeT : d;                                      \
    return _GFX_offload(&bilateral_noBorders_tiled<uchar, float, cn, depth, d, xCacheSizeT, yCacheSizeT, yBilatTiles>, srcptr, srcStep,    \
                                dstptr, dstStep,                                                                                          \
                                iters.x, iters.x + iters.width - 1, iters.y, iters.y + iters.height - 1,                                  \
                                gauss_color_coeff, maxK, space_weight, space_ofs);                                                        \
}                                                                                                                                         
            
            const int cn = 3;
            const int depth = 1;
            switch(windowSize) {
                case 3: inputCase(3, 32);
                case 5: inputCase(5, 32);
                case 7: inputCase(7, 32);
                case 9: inputCase(9, 32);
                case 11: inputCase(11, 32);
                default: 
                    onGpu_end(-1);
                    return -1;
            }
        }
        return -1;
    }

    static Size elementsPerIter(Mat m, int d) {
        const int xCacheSizeT = m.channels() * 32 + m.channels() * 2 * (d / 2);
        const int yCacheSizeT = ((2 * 1024 - d * d * 6) / xCacheSizeT > d) ? (2 * 1024 - d * d * 6) / xCacheSizeT : d;
        return Size(X_CACHE_EFFECTIVE_SIZE_PIX(xCacheSizeT, m.channels(), 1, d / 2),
                yBilatTiles * Y_CACHE_EFFECTIVE_SIZE_PIX(yCacheSizeT, m.channels(), 1, d / 2));
    }

    int onGpu_noBorders() {
        const Rect& iters = layout.gpuItersRect();
    
	    d_GFX_share(srcptr, getWholeMatSizeBytes(src));
	    d_GFX_share(dstptr, getWholeMatSizeBytes(dst));
	    d_GFX_share(space_weight, maxK * sizeof(float));
	    d_GFX_share(space_ofs, maxK * 2 * sizeof(int));
            
        const int cn = 3;
        const int depth = 1;
        return _GFX_offload(&bilateral_noBorders<uchar, float, cn, depth>, srcptr, srcStep,
                                    dstptr, dstStep,
                                    iters.x, iters.x + iters.width - 1, iters.y, iters.y + iters.height - 1,
                                    gauss_color_coeff, maxK, space_weight, space_ofs);
    }

    void onGpu_end(int k) {
        if (k >= 0)
            d_GFX_wait(k);
	    d_GFX_unshare(space_weight);
	    d_GFX_unshare(space_ofs);
	    d_GFX_unshare(srcptr);
	    d_GFX_unshare(dstptr);
    }

    void onCpu_atGpuBorder(int borderType, const void* borderConstantVal, bool extraExtrapolation) {
        onCpu_atGpuBorder_<uchar, float>(borderType, *((uchar*)borderConstantVal), extraExtrapolation);
    }
    
private:

    template< typename T, typename SUM_T >
    void onCpu_atGpuBorder_(int borderType, T borderConstantVal, bool extraExtrapolation) {
        const int leftBorderStart = layout.borderLeftTopGpuStart().x;
        const int topBorderStart = layout.borderLeftTopGpuStart().y;
        const int rightBorderStart = layout.borderRightBotGpuStart().x;
        const int bottomBorderStart = layout.borderRightBotGpuStart().y;
    
        const int leftBorderEnd = layout.borderLeftTopGpuEnd().x;
        const int topBorderEnd = layout.borderLeftTopGpuEnd().y;
        const int rightBorderEnd = layout.borderRightBotGpuEnd().x;
        const int bottomBorderEnd = layout.borderRightBotGpuEnd().y;

        const int leftAddrLimit = 0;
        const int topAddrLimit = 0;
        const int rightAddrLimit = cols - 1;
        const int bottomAddrLimit = rows - 1;

        Rect rois[4];
        computeCpuBorderRois(rois, layout);

        for (auto& roi : rois)
            if (roi.width && roi.height)
                bilateralFilter_onCpu(src(roi), dst(roi), windowSize, sigmaColor, sigmaSpace, borderType);

        //atBorder(bilateralFilter_atPixel(x, y, loadSrc_borderSafe, storeDst_direct));
    }


    int cn;
    int depth;

    const int cols;
    const int rows;

    uchar* srcptr;
    int srcStep;
    uchar* dstptr;
    int dstStep;

    const ImageOperationLayout& layout;

    const int windowSize;
    
    float sigmaColor;
    float sigmaSpace;
    float gauss_color_coeff;
    int maxK;
    float space_weight[17 * 17 * 4];
    char space_ofs[17 * 17 * 4 * 2];

    Mat& src;
    Mat& dst;
};


bool bilateral( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType ) {
    Mat src = _src.getMat();

    if( sigmaColor <= 0 )
        sigmaColor = 1;
    if( sigmaSpace <= 0 )
        sigmaSpace = 1;

    if( d <= 0 )
        d = max(cvRound(sigmaSpace * 1.5), 1) * 2 + 1;

    if (src.total() < (2900000 / d) || src.cols < d * 40 || src.rows < d * 40 || d % 2 != 1 || d > 11 || d < 3)
        return false;

    
    Mat dst = _dst.getMat();


    float gpuPart = 0.81f;
                                                                                                           
    if (src.type() == CV_8UC3) {                                                                                                             
        int r = d / 2;      

        ImageOperationLayout layout(Size(src.cols, src.rows), gpuPart, Size(src.elemSize(), 1), Size(src.elemSize(), 1), Bilateral::elementsPerIter(src, d), Size(r, r), true);
        Bilateral bilateral(src, dst, layout, d, sigmaColor, sigmaSpace);
        int tmp = 0;
        bilateral.onCpu_atGpuBorder(BORDER_CONSTANT, &tmp, false);
        int k = bilateral.onGpu_noBorders_tiled();
        if (k <= 0)
            return false;
        bilateralFilter_onCpu(src(layout.mainCpuWorkRect()), dst(layout.mainCpuWorkRect()), d, sigmaColor, sigmaSpace, borderType);
        bilateral.onGpu_end(k);                                                                                                        
    } else                                                                                                                                 
        return false;

    return true;
}


}
}
