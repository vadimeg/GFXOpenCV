#pragma once

#include "precomp.hpp"
#include <gfx/gfx_rt.h>
#include <cilk/cilk.h>
#include <algorithm>

namespace cv {
namespace igpu {

#define CV_IGPU_RUN(condition, func)              \
    if ((condition) && func)                      \
        return;


#define loadSrc_direct_greedy(x, y, value)                                                     \
        value = *(T*)(&srcptr[((y) * srcStep + (x)) * depth])
#define loadSrc_direct(x, y, value) {                                                          \
        if (depth == sizeof(T))                                                                \
            loadSrc_direct_greedy(x, y, value);                                                \
        else {                                                                                 \
            value = 0;                                                                         \
            for (int c = 0; c < depth; c++)                                                    \
                ((uchar*)&value)[c] = srcptr[((y) * srcStep + (x)) * depth + c];               \
        }                                                                                      \
}

#define loadSrc_tiled(x, y, value)                                                         \
        value = cachedSrc[y][x]
#define loadSrc_direct_tiled(xArg, yArg, value)                        \
        loadSrc_direct_greedy(xArg + X_CACHE_EFFECTIVE_SIZE_T_ * x, yArg + Y_CACHE_EFFECTIVE_SIZE_T_ * y, value)
#define storeDst_tiled(x, y, value)                                                         \
        cachedDst[y][x] = value;
#define storeDst_direct_tiled(xArg, yArg, value)                                                         \
        storeDst_direct(xArg + X_CACHE_EFFECTIVE_SIZE_T_ * x, yArg + Y_CACHE_EFFECTIVE_SIZE_T_ * y, value)

#define storeDst_direct(x, y, value)                                                               \
        if (depth == sizeof(T))                                                                \
            *(T*)(&dstptr[((y) * dstStep + (x)) * depth]) = value;                 \
        else                                                                                   \
            for (int c = 0; c < depth; c++)                                                    \
                dstptr[((y) * dstStep + (x)) * depth + c] = ((uchar*)&value)[c]

#define X_CACHE_SIZE_T(xCacheSizeT, cn, depth)  (xCacheSizeT)
#define Y_CACHE_SIZE_T(yCacheSizeT, cn, depth)  (yCacheSizeT)

#define X_CACHE_SIZE_T_  X_CACHE_SIZE_T(xCacheSizeT, cn, depth)
#define Y_CACHE_SIZE_T_  Y_CACHE_SIZE_T(yCacheSizeT, cn, depth)

#define X_CACHE_SIZE_PIX(xCacheSizeT, cn, depth, pixRadius)  (X_CACHE_SIZE_T(xCacheSizeT, cn, depth) / cn)
#define Y_CACHE_SIZE_PIX(yCacheSizeT, cn, depth, pixRadius)  (Y_CACHE_SIZE_T(yCacheSizeT, cn, depth))

#define X_CACHE_SIZE_PIX_  X_CACHE_SIZE_PIX(xCacheSizeT, cn, depth, xWindowSizePix / 2)
#define Y_CACHE_SIZE_PIX_  Y_CACHE_SIZE_PIX(yCacheSizeT, cn, depth, yWindowSizePix / 2)

#define X_CACHE_EFFECTIVE_SIZE_T(xCacheSizeT, cn, depth, pixRadius)  (X_CACHE_SIZE_T(xCacheSizeT, cn, depth) - 2 * cn * (pixRadius))
#define Y_CACHE_EFFECTIVE_SIZE_T(yCacheSizeT, cn, depth, pixRadius)  (Y_CACHE_SIZE_T(yCacheSizeT, cn, depth) - 2 * (pixRadius))

#define X_CACHE_EFFECTIVE_SIZE_PIX(xCacheSizeT, cn, depth, pixRadius)  (X_CACHE_SIZE_PIX(xCacheSizeT, cn, depth, pixRadius) - 2 * (pixRadius))
#define Y_CACHE_EFFECTIVE_SIZE_PIX(yCacheSizeT, cn, depth, pixRadius)  (Y_CACHE_SIZE_PIX(yCacheSizeT, cn, depth, pixRadius) - 2 * (pixRadius))


#define clamp(a, mi, ma) max(min(a, ma), mi)

#define extrapolate(x, minV, maxV) {                                                           \
    if (extraExtrapolation) {                                                                  \
        if ((borderType & BORDER_WRAP) == BORDER_WRAP) {                                       \
            if ((x) < (minV))                                                                  \
                (x) += ((maxV)-(minV));                                                        \
            if ((x) >= (maxV))                                                                 \
                (x) -= ((maxV)-(minV));                                                        \
        } if ((borderType & BORDER_REPLICATE) == BORDER_REPLICATE)                             \
            (x) = clamp((x), (minV), (maxV)-1);                                                \
        else if ((borderType & BORDER_REFLECT) == BORDER_REFLECT) {                            \
            if ((maxV)-(minV) == 1)                                                            \
                (x) = (minV);                                                                  \
            else                                                                               \
                while ((x) >= (maxV) || (x) < (minV))                                          \
                    if ((x) < (minV))                                                          \
                        (x) = (minV)-((x)-(minV)) - 1;                                         \
                    else                                                                       \
                        (x) = (maxV)-1 - ((x)-(maxV));                                         \
        } else if ((borderType & BORDER_REFLECT_101) == BORDER_REFLECT_101) {                  \
            if ((maxV)-(minV) == 1)                                                            \
                (x) = (minV);                                                                  \
            else                                                                               \
                while ((x) >= (maxV) || (x) < (minV)) {                                        \
                    if ((x) < (minV))                                                          \
                        (x) = (minV)-((x)-(minV));                                             \
                    else                                                                       \
                        (x) = (maxV)-1 - ((x)-(maxV)) - 1;                                     \
                }                                                                              \
        }                                                                                      \
    } else {                                                                                   \
        if ((borderType & BORDER_WRAP) == BORDER_WRAP) {                                       \
            if ((x) < (minV))                                                                  \
                (x) += (((minV)-(x)) / ((maxV)-(minV)) + 1) * ((maxV)-(minV));                 \
            if ((x) >= (maxV))                                                                 \
                (x) = ((x)-(minV)) % ((maxV)-(minV)) + (minV);                                 \
        } else if ((borderType & BORDER_REPLICATE) == BORDER_REPLICATE)                        \
            (x) = clamp((x), (minV), (maxV)-1);                                                \
        else if ((borderType & BORDER_REFLECT) == BORDER_REFLECT)                              \
            (x) = clamp((x), 2 * (minV) - (x) - 1, 2 * (maxV) - (x) - 1);                      \
        else if ((borderType & BORDER_REFLECT_101) == BORDER_REFLECT_101)                      \
            (x) = clamp((x), 2 * (minV) - (x), 2 * (maxV) - (x) - 2);                          \
    }                                                                                          \
}

#define loadSrc_borderSafe(x, y, value) {                                                                                                             \
            if (borderType == BORDER_CONSTANT) {                                                                                                      \
                value = borderConstantVal;                                                                                                                            \
            } else {                                                                                                                                  \
                int x_extrapolated;                                                                                                                   \
                int y_extrapolated = y;                                                                                                               \
                int x_offset;                                                                                                                         \
                if (x >= 0) {                                                                                                                         \
                    x_offset = (x) % cn;                                                                                                                \
                    x_extrapolated = (x) / cn;                                                                                                          \
                } else {                                                                                                                              \
                    x_offset = cn - (-x - 1) % cn - 1;                                                                                                \
                    x_extrapolated = (x - x_offset) / cn;                                                                                                \
                }                                                                                                                                     \
                extrapolate(x_extrapolated, leftAddrLimit, rightAddrLimit + 1);                                                                       \
                extrapolate(y_extrapolated, topAddrLimit, bottomAddrLimit + 1);                                                                       \
                x_extrapolated = x_extrapolated * cn + x_offset;                                                                                      \
                loadSrc_direct(x_extrapolated, y_extrapolated, value);                                                                                \
            }                                                                                                                                         \
}


#define atBorder(operation) {                                                                            \
    for (int y = topBorderStart; y < topBorderEnd; y++)                                                  \
        for (int x = leftBorderStart; x < rightBorderEnd; x++)                                           \
            operation;                                                                                   \
                                                                                                         \
    for (int y = topBorderEnd; y < bottomBorderStart; y++) {                                             \
        for (int x = leftBorderStart; x < leftBorderEnd; x++)                                            \
            operation;                                                                                   \
        for (int x = rightBorderStart; x < rightBorderEnd; x++)                                          \
            operation;                                                                                   \
    }                                                                                                    \
                                                                                                         \
    for (int y = bottomBorderStart; y < bottomBorderEnd; y++)                                            \
        for (int x = leftBorderStart; x < rightBorderEnd; x++)                                           \
            operation;                                                                                   \
}

  
#define atTile_initCache()                                                               \
        T cachedSrc[Y_CACHE_SIZE_T_][X_CACHE_SIZE_T_];


#define atTilePart_loadCache(x_block, y_block, tileStart, tileEnd)                                                                                \
            for (int yLocal = (tileStart); yLocal <= (tileEnd); yLocal++)                                                                      \
                memcpy(&(cachedSrc[yLocal][0]),                   \
                            srcptr + (((y_block) * Y_CACHE_EFFECTIVE_SIZE_T_ + yLocal) * srcStep + (x_block) * X_CACHE_EFFECTIVE_SIZE_T_) * depth,                                                                                      \
                            X_CACHE_SIZE_T_ * depth)

#define atTile_loadCache(x_block, y_block)                                                                                \
            atTilePart_loadCache(x_block, y_block, 0, Y_CACHE_SIZE_T_ - 1)


__declspec(target(gfx))
inline void memcpy_simple(void* restrict dst, const void* restrict src, int size) {
    for (int i = 0; i < size; i++)
        ((char*)(dst))[i] = ((char*)(src))[i];
}


class ImageOperationLayout {
public:

    /*_______________________
      |bbbbbbbbbbbbbbbbbbbbb|
      |bbbbbbb border bbbbbb|
      |bb gpu gpu gpu gpu bb|
      |bb gpu gpu gpu gpu bb|
      |bb gpu gpu gpu gpu bb|
      |bb gpu gpu gpu gpu bb|
      |bb  cache margin   bb|
      | cpu cpu cpu cpu cpu |
      | cpu cpu cpu cpu cpu |
      | cpu cpu cpu cpu cpu |
      -----------------------*/

    #define castX(si) (((si) * atomSizeBytes.width) / cpuAtomSizeBytes.width)
    #define castY(si) (((si) * atomSizeBytes.height) / cpuAtomSizeBytes.height)
    #define bcastX(si) (((si) * cpuAtomSizeBytes.width) / atomSizeBytes.width)
    #define bcastY(si) (((si) * cpuAtomSizeBytes.height) / atomSizeBytes.height)

    ImageOperationLayout(const Size& imageSize, float gpuWorkPart, const Size& cpuAtomSizeBytes, const Size& atomSizeBytes, const Size& atomsPerIteration, const Size& gpuBorderSize, bool isTiled) {
        Size gpuImageSize;

        if (gpuWorkPart < 0.f || gpuWorkPart > 1.f)
            throw 1;
        if (imageSize.height * (1.f - gpuWorkPart) < gpuBorderSize.height)
            gpuWorkPart = 1.f;

        gpuImageSize.width = imageSize.width;
        gpuImageSize.height = imageSize.height * gpuWorkPart;

        gpuIters.x = isTiled ? 0 : gpuBorderSize.width / atomsPerIteration.width;
        gpuIters.y = isTiled ? 0 : gpuBorderSize.height / atomsPerIteration.height;
        gpuIters.width = (gpuImageSize.width - 2 * gpuBorderSize.width) / atomsPerIteration.width;
        gpuIters.height = (gpuImageSize.height - 2 * gpuBorderSize.height) / atomsPerIteration.height;
        
        Point gpuStart;
        Point gpuEnd;
        Point mainCpuStart;
        Point mainCpuEnd;

        gpuStart.x = gpuBorderSize.width;
        gpuStart.y = gpuBorderSize.height;

        gpuEnd.x = gpuStart.x + gpuIters.width * atomsPerIteration.width;
        gpuEnd.y = gpuStart.y + gpuIters.height * atomsPerIteration.height;

        size_t rowSizeBytes = atomSizeBytes.height * atomSizeBytes.width * gpuImageSize.width;
        if (!rowSizeBytes)
            throw std::exception();

        const size_t igpuCacheLineSize = 128;
        size_t bottomCacheLineIsolationMargin = gpuEnd.y != (gpuImageSize.height - 1) ? 
            igpuCacheLineSize / rowSizeBytes + !!(128 % rowSizeBytes) :
            0;

        mainCpuStart.x = 0;
        mainCpuStart.y = castY(gpuEnd.y) + bottomCacheLineIsolationMargin;
        mainCpuEnd.x = castX(imageSize.width);
        mainCpuEnd.y = castY(imageSize.height);
        
        //cout << gpuStart << " " << gpuEnd << endl;
        //cout << gpuIters << endl;
        //cout << mainCpuStart << " " << mainCpuEnd << endl;

        mainCpuWorkRect_ = Rect(mainCpuStart, mainCpuEnd);

        
        leftTopStartCpu.x = 0;
        leftTopStartCpu.y = 0;
        leftTopEndCpu.x = castX(gpuStart.x);
        leftTopEndCpu.y = castY(gpuStart.y);
        
        rightBotStartCpu.x = castX(imageSize.width) - castX(imageSize.width - gpuEnd.x);
        rightBotStartCpu.y = castY(imageSize.height) - castY(imageSize.height - gpuEnd.y);
        rightBotEndCpu.x = castX(imageSize.width);
        rightBotEndCpu.y = rightBotStartCpu.y + bcastY(bottomCacheLineIsolationMargin);
        
        leftTopStartGpu.x = bcastX(leftTopEndCpu.x);
        leftTopStartGpu.y = bcastY(leftTopEndCpu.y);
        leftTopEndGpu.x = gpuStart.x;
        leftTopEndGpu.y = gpuStart.y;
    
        rightBotStartGpu.x = gpuEnd.x;
        rightBotStartGpu.y = gpuEnd.y + bottomCacheLineIsolationMargin;
        rightBotEndGpu.x = bcastX(rightBotStartCpu.x);
        rightBotEndGpu.y = bcastY(rightBotStartCpu.y);
    }

    const Rect& gpuItersRect() const {
        return gpuIters;
    }

    const Rect& mainCpuWorkRect() const {
        return mainCpuWorkRect_;
    }


    const Point& borderLeftTopCpuStart() const {
        return leftTopStartCpu;
    }

    const Point& borderLeftTopCpuEnd() const {
        return leftTopEndCpu;
    }

    const Point& borderRightBotCpuStart() const {
        return rightBotStartCpu;
    }

    const Point& borderRightBotCpuEnd() const {
        return rightBotEndCpu;
    }


    const Point& borderLeftTopGpuStart() const {
        return leftTopStartGpu;
    }

    const Point& borderLeftTopGpuEnd() const {
        return leftTopEndGpu;
    }

    const Point& borderRightBotGpuStart() const {
        return rightBotStartGpu;
    }

    const Point& borderRightBotGpuEnd() const {
        return rightBotEndGpu;
    }

private:
    Point leftTopStartCpu;
    Point leftTopEndCpu;
    Point rightBotStartCpu;
    Point rightBotEndCpu;

    Point leftTopStartGpu;
    Point leftTopEndGpu;
    Point rightBotStartGpu;
    Point rightBotEndGpu;

    Rect gpuIters;
    Rect mainCpuWorkRect_;
};

void static computeCpuBorderRois(Rect* rois, const ImageOperationLayout& layout) {
        rois[0] = Rect(layout.borderLeftTopCpuStart(), Point(layout.borderRightBotCpuEnd().x, layout.borderLeftTopCpuEnd().y));
        rois[1] = Rect(Point(0, layout.borderLeftTopCpuEnd().y), Point(layout.borderLeftTopCpuEnd().x, layout.borderRightBotCpuStart().y));
        rois[2] = Rect(Point(layout.borderRightBotCpuStart().x, layout.borderLeftTopCpuEnd().y), Point(layout.borderRightBotCpuEnd().x, layout.borderRightBotCpuStart().y));
        rois[3] = Rect(Point(0, layout.borderRightBotCpuStart().y), layout.borderRightBotCpuEnd());
}

template< class MatT >
static bool makeKernelSquare(MatT& kernel) throw(int) {
    if (fmod(kernel.total() * kernel.channels(), 1) < 0.0001) {
        int sizes[2];
        sizes[0] = sizes[1] = sqrt(kernel.total() * kernel.channels());
        kernel = kernel.reshape(1, 2, sizes);
        if (!kernel.isContinuous())
            kernel = kernel.clone();
        return true;
    } else {
        std::cout << "some error at " << __LINE__ << " in " << __FILE__ << std::endl;
        return false;
    }
}

template< class MatT >
static size_t getWholeMatSizeBytes(MatT& mat) throw(int) {
    Point tmp;
    Size wholeSize;
    mat.locateROI(wholeSize, tmp);
    return wholeSize.height * wholeSize.width * mat.elemSize();
}

template< class MatT >
static void testContin(MatT& mat) throw(int) {
    if (!mat.isContinuous())
        cout << "!mat.isContinious()" << endl;
}


#define te(expr) expr

#define d_GFX_share(expr1, expr2) te(_GFX_share(expr1, expr2))
#define d_GFX_unshare(expr) te(_GFX_unshare(expr))
#define d_GFX_wait(expr) te(_GFX_wait(expr))

}
}


#define tstart start = std::chrono::steady_clock::now();

#define tend(msg)                                                       \
duration = std::chrono::duration_cast< std::chrono::milliseconds >       \
		(std::chrono::system_clock::now() - start);                           \
	std::cout << msg << "   " << duration.count() << std::endl;
#define t2start start2 = std::chrono::steady_clock::now();

#define t2end(msg)                                                       \
duration = std::chrono::duration_cast< std::chrono::milliseconds >       \
		(std::chrono::system_clock::now() - start2);                           \
	std::cout << msg << "   " << duration.count() << std::endl;


static auto start = std::chrono::system_clock::now();
static auto start2 = std::chrono::system_clock::now();
static auto duration = std::chrono::duration_cast< std::chrono::milliseconds >
	(std::chrono::system_clock::now() - start);
