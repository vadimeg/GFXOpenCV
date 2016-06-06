#pragma once

#include <gfx/gfx_rt.h>
#include <cilk/cilk.h>

namespace cv {
namespace igpu {

#define unrolled_for __pragma(unroll) for
#define simd_for __pragma(simd) for
#define unrolled_simd_for __pragma(simd) unrolled_for
#define simd_cilk_for __pragma(simd) cilk_for

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
#define storeDst_rowCached(x, y, value)                                                         \
        cachedDstRow[x - dstXCacheOffset] = value
#define loadSrc_direct_tiled(xArg, yArg, value)                                               \
        loadSrc_direct_greedy(xArg + xTileTSize * x, yArg + yTileTSize * y, value)
#define storeDst_tiled(x, y, value)                                                           \
        cachedDst[y][x] = value;
#define storeDst_direct_tiled(xArg, yArg, value)                                              \
        storeDst_direct(xArg + xTileTSize * x, yArg + yTileTSize * y, value)

#define storeDst_direct(x, y, value)                                                            \
        if (depth == sizeof(T))                                                                \
            *(T*)(&dstptr[((y) * dstStep + (x)) * depth]) = value;                 \
        else                                                                                   \
            for (int c = 0; c < depth; c++)                                                    \
                dstptr[((y) * dstStep + (x)) * depth + c] = ((uchar*)&value)[c]



#define static_abs(a) ((a) >= 0 ? (a) : -(a))
#define static_max(a, b) ((a) > (b) ? (a) : (b))
#define static_min(a, b) ((a) < (b) ? (a) : (b))
#define clamp(a, mi, ma) static_max(static_min(a, ma), mi)


#define atTile_initCache()                                                               \
        T cachedSrc[yCacheTSize][xCacheTSize];


//#define atTilePart_loadCache(xTile, yTile, xOff, yOff, tileStart, tileEnd)                                                                                \
//        for (int yLocal = (tileStart); yLocal <= (tileEnd); yLocal++)                                                                      \
//            memcpy_simple(&(cachedSrc[yLocal][0]),                   \
//                        srcptr + (((yTile) * yTileTSize + yLocal - (yOff)) * srcStep + (xTile) * xTileTSize - (xOff)),           \
//                        xCacheTSize)

#define atTilePart_loadCache(xTile, yTile, xOff, yOff, tileStart, tileEnd) {                                                                \
        const int yLen = (tileEnd) - (tileStart) + 1;                                                                                           \
        cachedSrc[(tileStart) : yLen][:] =                                                                                                  \
                            src[((yTile) * yTileTSize + (tileStart) - (yOff)) : yLen][((xTile) * xTileTSize - (xOff)) : xCacheTSize];            \
}



#define atTile_loadCache(xTile, yTile, xOff, yOff)                                                                                \
            atTilePart_loadCache(xTile, yTile, xOff, yOff, 0, yCacheTSize - 1)


static const int igpuRegisterFileSize = 2 * 1024/* + 512*/;


__declspec(target(gfx))
inline void memcpy_simple(void* restrict dst, const void* restrict src, int size) {
    for (int i = 0; i < size; i++)
        ((char*)(dst))[i] = ((char*)(src))[i];
}

__declspec(target(gfx))
inline void memcpy_(void* restrict dst, const void* restrict src, int size) {
    const int s = size / 4;
    simd_for (int i = 0; i < s; i++)
        ((int*)(dst))[i] = ((int*)(src))[i];
    for (int i = s * 4; i < size; i++)
        ((char*)(dst))[i] = ((char*)(src))[i];
}

}
}

