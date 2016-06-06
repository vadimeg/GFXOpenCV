#pragma once

#include <gfx/gfx_rt.h>
#include <cilk/cilk.h>

namespace cv {
namespace igpu {

#define unrolled_for __pragma(unroll) for
#define simd_for __pragma(simd) for
#define unrolled_simd_for __pragma(simd) unrolled_for
#define simd_cilk_for __pragma(simd) cilk_for

#define static_abs(a) ((a) >= 0 ? (a) : -(a))
#define static_max(a, b) ((a) > (b) ? (a) : (b))
#define static_min(a, b) ((a) < (b) ? (a) : (b))
#define clamp(a, mi, ma) static_max(static_min(a, ma), mi)

#define atTile_initCache()                                                               \
        T cachedSrc[yCacheTSize][xCacheTSize];

//#define atTilePart_loadCache(xTile, yTile, xOff, yOff, tileStart, tileEnd)                                                                                \
//        for (int y__ = (tileStart); y__ <= (tileEnd); y__++)                                                                      \
//            memcpy_(&(cachedSrc[y__][0]),                                                                                     \
//                        srcptr + (((yTile) * yTileTSize + y__ - (yOff)) * srcStep + (xTile) * xTileTSize - (xOff)),           \
//                        xCacheTSize)

#define atTilePart_loadCache(xTile, yTile, xOff, yOff, tileStart, tileEnd) {                                                                \
        const int yLen = (tileEnd) - (tileStart) + 1;                                                                                           \
        cachedSrc[(tileStart) : yLen][:] =                                                                                                  \
                            src[((yTile) * yTileTSize + (tileStart) - (yOff)) : yLen][((xTile) * xTileTSize - (xOff)) : xCacheTSize];            \
}

//#define atTilePart_loadCache(xTile, yTile, xOff, yOff, tileStart, tileEnd) {                                                                \
//        for (int y__ = (tileStart); y__ <= (tileEnd); y__++)                                                                                        \
//            cachedSrc[(tileStart) + y__][:] =                                                                                                  \
//                                src[((yTile) * yTileTSize + (tileStart) - (yOff)) + y__][((xTile) * xTileTSize - (xOff)) : xCacheTSize];            \
//}

#define atTile_loadCache(xTile, yTile, xOff, yOff)                                                                                \
            atTilePart_loadCache(xTile, yTile, xOff, yOff, 0, yCacheTSize - 1)


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

